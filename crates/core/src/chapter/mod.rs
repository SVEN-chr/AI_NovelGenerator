use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::architecture::{
    LanguageModel, LanguageModelError, ARCHITECTURE_FILE_NAME, CHARACTER_STATE_FILE_NAME,
};
use crate::blueprint::{ChapterBlueprint, ChapterBlueprintEntry};
use crate::logging::{LogLevel, LogRecord, LogSink};
use crate::prompts::{PromptError, PromptRegistry};

mod keywords;
mod knowledge;
mod prompt;
mod summary;

use keywords::generate_keyword_groups;
use knowledge::{apply_content_rules, filter_knowledge_contexts};
use prompt::{render_first_chapter_prompt, render_next_chapter_prompt};
use summary::{extract_summary, load_recent_chapters, summarize_recent_chapters};

const GLOBAL_SUMMARY_FILE_NAME: &str = "global_summary.txt";
const CHAPTERS_DIR_NAME: &str = "chapters";
const MAX_HISTORY_CHAPTERS: usize = 3;
const MAX_SUMMARY_SOURCE_CHARS: usize = 4_000;
const MAX_SUMMARY_OUTPUT_CHARS: usize = 2_000;
const PREVIOUS_EXCERPT_CHARS: usize = 800;
const KNOWLEDGE_SNIPPET_MAX_CHARS: usize = 600;
const KNOWLEDGE_FALLBACK: &str = "（知识库处理失败）";
const KNOWLEDGE_EMPTY: &str = "（无相关知识库内容）";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChapterStage {
    Summary,
    KeywordGeneration,
    KnowledgeFilter,
    Prompt,
    Draft,
}

impl ChapterStage {
    fn label(&self) -> &'static str {
        match self {
            Self::Summary => "章节摘要",
            Self::KeywordGeneration => "关键词检索",
            Self::KnowledgeFilter => "知识过滤",
            Self::Prompt => "提示词构建",
            Self::Draft => "章节草稿生成",
        }
    }
}

impl fmt::Display for ChapterStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug)]
pub struct KnowledgeBaseError {
    inner: Box<dyn std::error::Error + Send + Sync>,
}

impl KnowledgeBaseError {
    pub fn new<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            inner: Box::new(error),
        }
    }

    pub fn into_inner(self) -> Box<dyn std::error::Error + Send + Sync> {
        self.inner
    }
}

impl fmt::Display for KnowledgeBaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::error::Error for KnowledgeBaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.inner.as_ref())
    }
}

pub trait KnowledgeBase: Send + Sync {
    fn count(&self) -> Result<Option<usize>, KnowledgeBaseError> {
        Ok(None)
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<String>, KnowledgeBaseError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ChapterError {
    #[error("未在章节蓝图中找到第{number}章信息")]
    MissingChapter { number: u32 },
    #[error("无法创建目录 `{path}`: {source}")]
    CreateDir { path: PathBuf, source: io::Error },
    #[error("读取文件 `{path}` 失败: {source}")]
    ReadFile { path: PathBuf, source: io::Error },
    #[error("写入文件 `{path}` 失败: {source}")]
    WriteFile { path: PathBuf, source: io::Error },
    #[error("渲染{stage}提示词失败: {source}")]
    Prompt {
        stage: ChapterStage,
        #[source]
        source: PromptError,
    },
    #[error("调用模型执行{stage}失败: {source}")]
    Model {
        stage: ChapterStage,
        #[source]
        source: LanguageModelError,
    },
    #[error("向量检索失败: {source}")]
    KnowledgeBase {
        #[source]
        source: KnowledgeBaseError,
    },
}

#[derive(Clone, Debug)]
pub struct ChapterPromptRequest<'a> {
    pub output_dir: PathBuf,
    pub blueprint: &'a ChapterBlueprint,
    pub novel_number: u32,
    pub word_number: u32,
    pub user_guidance: String,
    pub characters_involved: String,
    pub key_items: String,
    pub scene_location: String,
    pub time_constraint: String,
    pub embedding_retrieval_k: usize,
    pub history_chapter_count: usize,
}

impl<'a> ChapterPromptRequest<'a> {
    pub fn new(
        output_dir: impl Into<PathBuf>,
        blueprint: &'a ChapterBlueprint,
        novel_number: u32,
        word_number: u32,
    ) -> Self {
        Self {
            output_dir: output_dir.into(),
            blueprint,
            novel_number,
            word_number,
            user_guidance: String::new(),
            characters_involved: String::new(),
            key_items: String::new(),
            scene_location: String::new(),
            time_constraint: String::new(),
            embedding_retrieval_k: 2,
            history_chapter_count: MAX_HISTORY_CHAPTERS,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChapterPrompt {
    pub prompt_text: String,
    pub summary: String,
    pub keyword_groups: Vec<String>,
    pub knowledge_contexts: Vec<String>,
    pub filtered_context: String,
}

#[derive(Clone, Debug)]
pub struct ChapterDraft {
    pub chapter_number: u32,
    pub content: String,
    pub prompt: String,
    pub path: PathBuf,
    pub summary: String,
    pub filtered_context: String,
}

pub struct ChapterService<'a> {
    prompts: &'a PromptRegistry,
    sink: &'a dyn LogSink,
}

impl<'a> ChapterService<'a> {
    pub fn new(prompts: &'a PromptRegistry, sink: &'a dyn LogSink) -> Self {
        Self { prompts, sink }
    }

    pub fn build_chapter_prompt<M: LanguageModel, K: KnowledgeBase>(
        &self,
        model: &M,
        knowledge_base: Option<&K>,
        request: &ChapterPromptRequest<'_>,
    ) -> Result<ChapterPrompt, ChapterError> {
        let chapter = request.blueprint.chapter(request.novel_number).ok_or(
            ChapterError::MissingChapter {
                number: request.novel_number,
            },
        )?;

        if request.novel_number == 1 {
            return self.render_first_chapter_prompt(chapter, request);
        }

        let next_chapter = request
            .novel_number
            .checked_add(1)
            .and_then(|next| request.blueprint.chapter(next));

        let global_summary =
            read_optional_file(&request.output_dir.join(GLOBAL_SUMMARY_FILE_NAME))?;
        let character_state =
            read_optional_file(&request.output_dir.join(CHARACTER_STATE_FILE_NAME))?;

        let chapters_dir = request.output_dir.join(CHAPTERS_DIR_NAME);
        let history_count = request.history_chapter_count.max(1);
        let recent_chapters =
            load_recent_chapters(&chapters_dir, request.novel_number, history_count);
        let combined_text = recent_chapters.join("\n");
        let combined_text_tail = tail_chars(&combined_text, MAX_SUMMARY_SOURCE_CHARS);

        let mut summary_text = String::new();
        if !combined_text_tail.trim().is_empty() {
            self.log(LogLevel::Info, "生成章节摘要");
            let summary_response = summarize_recent_chapters(
                model,
                self.prompts,
                chapter,
                next_chapter,
                combined_text_tail,
                request.novel_number,
            )?;
            summary_text = extract_summary(&summary_response);
            if summary_text.is_empty() {
                summary_text = truncate_to_owned(summary_response.trim(), MAX_SUMMARY_OUTPUT_CHARS);
            } else {
                summary_text = truncate_to_owned(&summary_text, MAX_SUMMARY_OUTPUT_CHARS);
            }
        }

        let previous_excerpt = recent_chapters
            .iter()
            .rev()
            .find(|text| !text.trim().is_empty())
            .map(|text| tail_to_owned(text, PREVIOUS_EXCERPT_CHARS))
            .unwrap_or_default();

        self.log(LogLevel::Info, "生成知识检索关键词");
        let keyword_groups =
            generate_keyword_groups(model, self.prompts, chapter, request, &summary_text)?;

        let mut knowledge_contexts = Vec::new();
        if let Some(base) = knowledge_base {
            let requested_k = request.embedding_retrieval_k.max(1);
            let available_k = match base.count() {
                Ok(option) => option
                    .filter(|count| *count > 0)
                    .map(|count| count.min(requested_k))
                    .unwrap_or(requested_k),
                Err(err) => {
                    self.log(LogLevel::Error, format!("获取向量库容量失败: {err}"));
                    requested_k
                }
            };

            for group in &keyword_groups {
                match base.search(group, available_k) {
                    Ok(chunks) => {
                        for chunk in chunks {
                            if chunk.trim().is_empty() {
                                continue;
                            }
                            let label = classify_keyword_group(group);
                            knowledge_contexts.push(format!("{label} {chunk}"));
                        }
                    }
                    Err(err) => {
                        self.log(LogLevel::Error, format!("知识检索失败（{group}）: {err}"));
                    }
                }
            }
        }

        let processed_contexts = apply_content_rules(&knowledge_contexts, request.novel_number);
        let filtered_context = if processed_contexts.is_empty() {
            KNOWLEDGE_EMPTY.to_string()
        } else {
            match filter_knowledge_contexts(
                model,
                self.prompts,
                chapter,
                request,
                &processed_contexts,
            ) {
                Ok(text) => text,
                Err(err) => {
                    self.log(LogLevel::Error, format!("知识过滤失败: {err}"));
                    KNOWLEDGE_FALLBACK.to_string()
                }
            }
        };

        let prompt_text = render_next_chapter_prompt(
            self.prompts,
            chapter,
            next_chapter,
            request,
            &global_summary,
            &previous_excerpt,
            &character_state,
            &summary_text,
            &filtered_context,
        )?;

        Ok(ChapterPrompt {
            prompt_text,
            summary: summary_text,
            keyword_groups,
            knowledge_contexts: processed_contexts,
            filtered_context,
        })
    }

    pub fn generate_chapter_draft<M: LanguageModel, K: KnowledgeBase>(
        &self,
        model: &M,
        knowledge_base: Option<&K>,
        request: &ChapterPromptRequest<'_>,
        custom_prompt: Option<&str>,
    ) -> Result<ChapterDraft, ChapterError> {
        let prompt_result = if let Some(text) = custom_prompt {
            ChapterPrompt {
                prompt_text: text.to_string(),
                ..ChapterPrompt::default()
            }
        } else {
            self.build_chapter_prompt(model, knowledge_base, request)?
        };

        self.log(
            LogLevel::Info,
            format!("调用模型生成第{}章草稿", request.novel_number),
        );
        let response =
            model
                .invoke(&prompt_result.prompt_text)
                .map_err(|source| ChapterError::Model {
                    stage: ChapterStage::Draft,
                    source,
                })?;

        let chapters_dir = request.output_dir.join(CHAPTERS_DIR_NAME);
        fs::create_dir_all(&chapters_dir).map_err(|source| ChapterError::CreateDir {
            path: chapters_dir.clone(),
            source,
        })?;

        let chapter_path = chapters_dir.join(format!("chapter_{}.txt", request.novel_number));
        fs::write(&chapter_path, response.as_bytes()).map_err(|source| {
            ChapterError::WriteFile {
                path: chapter_path.clone(),
                source,
            }
        })?;

        Ok(ChapterDraft {
            chapter_number: request.novel_number,
            content: response,
            prompt: prompt_result.prompt_text,
            path: chapter_path,
            summary: prompt_result.summary,
            filtered_context: prompt_result.filtered_context,
        })
    }

    fn render_first_chapter_prompt(
        &self,
        chapter: &ChapterBlueprintEntry,
        request: &ChapterPromptRequest<'_>,
    ) -> Result<ChapterPrompt, ChapterError> {
        let architecture = read_optional_file(&request.output_dir.join(ARCHITECTURE_FILE_NAME))?;
        let prompt_text =
            render_first_chapter_prompt(self.prompts, chapter, request, &architecture)?;
        Ok(ChapterPrompt {
            prompt_text,
            ..ChapterPrompt::default()
        })
    }

    fn log(&self, level: LogLevel, message: impl Into<String>) {
        self.sink.log(LogRecord::new(level, message.into()));
    }
}

fn read_optional_file(path: &Path) -> Result<String, ChapterError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(source) => Err(ChapterError::ReadFile {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn tail_chars<'a>(text: &'a str, max_chars: usize) -> &'a str {
    if max_chars == 0 {
        return "";
    }
    let mut count = 0usize;
    for (idx, _) in text.char_indices().rev() {
        count += 1;
        if count == max_chars {
            return &text[idx..];
        }
    }
    text
}

fn truncate_to_owned(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut end = text.len();
    for (index, (idx, _)) in text.char_indices().enumerate() {
        if index == max_chars {
            end = idx;
            break;
        }
    }
    if end == text.len() {
        text.to_string()
    } else {
        text[..end].to_string()
    }
}

fn tail_to_owned(text: &str, max_chars: usize) -> String {
    tail_chars(text, max_chars).to_string()
}

fn classify_keyword_group(group: &str) -> &'static str {
    let normalized = group.to_lowercase();
    if ["技法", "手法", "模板"].iter().any(|kw| group.contains(kw)) {
        "[TECHNIQUE]"
    } else if ["设定", "技术", "世界观"]
        .iter()
        .any(|kw| group.contains(kw))
    {
        "[SETTING]"
    } else if normalized.contains("technique") {
        "[TECHNIQUE]"
    } else if normalized.contains("setting") {
        "[SETTING]"
    } else {
        "[GENERAL]"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_keyword_groups() {
        assert_eq!(classify_keyword_group("悬念技法"), "[TECHNIQUE]");
        assert_eq!(classify_keyword_group("科技设定"), "[SETTING]");
        assert_eq!(classify_keyword_group("普通 描写"), "[GENERAL]");
    }
}
