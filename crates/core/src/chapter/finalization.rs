use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::architecture::{LanguageModel, LanguageModelError, CHARACTER_STATE_FILE_NAME};
use crate::embedding::EmbeddingModel;
use crate::logging::{LogLevel, LogRecord, LogSink};
use crate::prompts::{PromptArguments, PromptError, PromptRegistry};

const GLOBAL_SUMMARY_FILE_NAME: &str = "global_summary.txt";
const CHAPTERS_DIR_NAME: &str = "chapters";
const MAX_RETRIES: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinalizeStage {
    Summary,
    CharacterState,
    Enrichment,
}

impl FinalizeStage {
    fn label(&self) -> &'static str {
        match self {
            Self::Summary => "前文摘要更新",
            Self::CharacterState => "角色状态更新",
            Self::Enrichment => "章节扩写",
        }
    }
}

impl std::fmt::Display for FinalizeStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Error)]
pub enum FinalizeError {
    #[error("未找到第{chapter_number}章的草稿文件")]
    MissingChapter { chapter_number: u32 },
    #[error("第{chapter_number}章内容为空，无法完成定稿")]
    EmptyChapter { chapter_number: u32 },
    #[error("读取文件 `{path}` 失败: {source}")]
    ReadFile { path: PathBuf, source: io::Error },
    #[error("写入文件 `{path}` 失败: {source}")]
    WriteFile { path: PathBuf, source: io::Error },
    #[error("渲染{stage}提示词失败: {source}")]
    Prompt {
        stage: FinalizeStage,
        #[source]
        source: PromptError,
    },
    #[error("调用模型执行{stage}失败: {source}")]
    Model {
        stage: FinalizeStage,
        #[source]
        source: LanguageModelError,
    },
}

#[derive(Clone, Debug)]
pub struct FinalizeChapterRequest {
    pub output_dir: PathBuf,
    pub chapter_number: u32,
}

#[derive(Clone, Debug)]
pub struct FinalizeChapterResult {
    pub chapter_path: PathBuf,
    pub summary_path: PathBuf,
    pub character_state_path: PathBuf,
    pub summary_text: String,
    pub character_state_text: String,
    pub segments_written: usize,
}

#[derive(Clone, Debug)]
pub struct EnrichChapterRequest<'a> {
    pub chapter_number: Option<u32>,
    pub chapter_text: &'a str,
    pub target_word_count: u32,
}

pub struct ChapterFinalizer<'a> {
    prompts: &'a PromptRegistry,
    sink: &'a dyn LogSink,
    max_retries: usize,
}

impl<'a> ChapterFinalizer<'a> {
    pub fn new(prompts: &'a PromptRegistry, sink: &'a dyn LogSink) -> Self {
        Self {
            prompts,
            sink,
            max_retries: MAX_RETRIES,
        }
    }

    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries.max(1);
        self
    }

    pub fn finalize_chapter<M: LanguageModel + ?Sized>(
        &self,
        model: &M,
        _embedding: Option<&dyn EmbeddingModel>,
        request: &FinalizeChapterRequest,
    ) -> Result<FinalizeChapterResult, FinalizeError> {
        let chapter_path = request
            .output_dir
            .join(CHAPTERS_DIR_NAME)
            .join(format!("chapter_{}.txt", request.chapter_number));

        let chapter_text = read_required_file(&chapter_path).map_err(|err| match err {
            IoAccessError::NotFound => FinalizeError::MissingChapter {
                chapter_number: request.chapter_number,
            },
            IoAccessError::Io { path, source } => FinalizeError::ReadFile { path, source },
        })?;

        let trimmed = chapter_text.trim();
        if trimmed.is_empty() {
            return Err(FinalizeError::EmptyChapter {
                chapter_number: request.chapter_number,
            });
        }

        let summary_path = request.output_dir.join(GLOBAL_SUMMARY_FILE_NAME);
        let character_state_path = request.output_dir.join(CHARACTER_STATE_FILE_NAME);

        let old_summary = read_optional_file(&summary_path)?;
        let old_character_state = read_optional_file(&character_state_path)?;

        self.log(
            LogLevel::Info,
            format!(
                "开始定稿第{}章，更新前文摘要与角色状态。",
                request.chapter_number
            ),
        );

        let mut args = PromptArguments::new();
        args.insert("chapter_text".into(), trimmed.to_string());
        args.insert("global_summary".into(), old_summary.clone());
        let summary_prompt =
            self.prompts
                .format("summary", &args)
                .map_err(|source| FinalizeError::Prompt {
                    stage: FinalizeStage::Summary,
                    source,
                })?;

        let new_summary =
            self.invoke_with_cleaning(model, FinalizeStage::Summary, &summary_prompt)?;
        let summary_to_write = if new_summary.trim().is_empty() {
            self.log(LogLevel::Warn, "摘要更新返回空文本，保留原有前文摘要。");
            old_summary.clone()
        } else {
            new_summary.trim().to_string()
        };

        let mut args = PromptArguments::new();
        args.insert("chapter_text".into(), trimmed.to_string());
        args.insert("old_state".into(), old_character_state.clone());
        let character_prompt = self
            .prompts
            .format("update_character_state", &args)
            .map_err(|source| FinalizeError::Prompt {
                stage: FinalizeStage::CharacterState,
                source,
            })?;

        let new_character_state =
            self.invoke_with_cleaning(model, FinalizeStage::CharacterState, &character_prompt)?;
        let character_state_to_write = if new_character_state.trim().is_empty() {
            self.log(LogLevel::Warn, "角色状态更新返回空文本，保留原有角色状态。");
            old_character_state.clone()
        } else {
            new_character_state.trim().to_string()
        };

        truncate_file(&summary_path).map_err(|source| FinalizeError::WriteFile {
            path: summary_path.clone(),
            source,
        })?;
        write_string(&summary_path, &summary_to_write).map_err(|source| {
            FinalizeError::WriteFile {
                path: summary_path.clone(),
                source,
            }
        })?;

        truncate_file(&character_state_path).map_err(|source| FinalizeError::WriteFile {
            path: character_state_path.clone(),
            source,
        })?;
        write_string(&character_state_path, &character_state_to_write).map_err(|source| {
            FinalizeError::WriteFile {
                path: character_state_path.clone(),
                source,
            }
        })?;

        self.log(
            LogLevel::Info,
            format!(
                "第{}章定稿完成。摘要与角色状态已更新。",
                request.chapter_number
            ),
        );

        Ok(FinalizeChapterResult {
            chapter_path,
            summary_path,
            character_state_path,
            summary_text: summary_to_write,
            character_state_text: character_state_to_write,
            segments_written: 0,
        })
    }

    pub fn enrich_chapter_text<M: LanguageModel>(
        &self,
        model: &M,
        request: &EnrichChapterRequest<'_>,
    ) -> Result<String, FinalizeError> {
        let stage = FinalizeStage::Enrichment;
        let prompt = format!(
            "以下章节文本较短，请在保持剧情连贯的前提下进行扩写，使其更充实，接近 {} 字左右：\n原内容：\n{}",
            request.target_word_count,
            request.chapter_text
        );

        if let Some(number) = request.chapter_number {
            self.log(LogLevel::Info, format!("开始扩写第{}章文本。", number));
        } else {
            self.log(LogLevel::Info, "开始扩写章节文本。".to_string());
        }

        let result = self.invoke_with_cleaning(model, stage, &prompt)?;
        if result.trim().is_empty() {
            self.log(LogLevel::Warn, "扩写模型返回空文本，保留原始内容。");
            Ok(request.chapter_text.to_string())
        } else {
            Ok(result)
        }
    }

    fn invoke_with_cleaning<M: LanguageModel + ?Sized>(
        &self,
        model: &M,
        stage: FinalizeStage,
        prompt: &str,
    ) -> Result<String, FinalizeError> {
        let mut last_response = String::new();

        for attempt in 1..=self.max_retries {
            self.log(
                LogLevel::Info,
                format!(
                    "发送到 LLM 的提示词（{}｜第{}次尝试）：\n{}",
                    stage.label(),
                    attempt,
                    prompt
                ),
            );

            match model.invoke(prompt) {
                Ok(response) => {
                    self.log(
                        LogLevel::Info,
                        format!(
                            "LLM 返回的内容（{}｜第{}次尝试）：\n{}",
                            stage.label(),
                            attempt,
                            response
                        ),
                    );
                    let cleaned = response.replace("```", "").trim().to_string();
                    if !cleaned.is_empty() {
                        return Ok(cleaned);
                    }
                    last_response = cleaned;
                    self.log(
                        LogLevel::Warn,
                        format!(
                            "LLM 返回空响应，准备重试（{}｜第{}次尝试）",
                            stage.label(),
                            attempt
                        ),
                    );
                }
                Err(err) => {
                    self.log(
                        LogLevel::Error,
                        format!(
                            "模型调用失败（{}｜第{}次尝试）：{err}",
                            stage.label(),
                            attempt
                        ),
                    );
                    if attempt == self.max_retries {
                        return Err(FinalizeError::Model { stage, source: err });
                    }
                }
            }
        }

        Ok(last_response)
    }

    fn log(&self, level: LogLevel, message: impl Into<String>) {
        self.sink.log(LogRecord::new(level, message.into()));
    }
}

enum IoAccessError {
    NotFound,
    Io { path: PathBuf, source: io::Error },
}

fn read_required_file(path: &Path) -> Result<String, IoAccessError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Err(IoAccessError::NotFound),
        Err(source) => Err(IoAccessError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn read_optional_file(path: &Path) -> Result<String, FinalizeError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(source) => Err(FinalizeError::ReadFile {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn truncate_file(path: &Path) -> Result<(), io::Error> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    Ok(())
}

fn write_string(path: &Path, content: &str) -> Result<(), io::Error> {
    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
    file.write_all(content.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::LanguageModel;
    use crate::logging::NullLogSink;
    use crate::prompts::PromptRegistry;
    use tempfile::tempdir;

    struct MockLanguageModel {
        responses: Vec<String>,
    }

    impl MockLanguageModel {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.into_iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl LanguageModel for MockLanguageModel {
        fn invoke(&self, _prompt: &str) -> Result<String, LanguageModelError> {
            Ok(self.responses.first().cloned().unwrap_or_else(String::new))
        }
    }

    #[test]
    fn enrich_returns_original_when_empty() {
        let registry = PromptRegistry::new().unwrap();
        let sink = NullLogSink;
        let finalizer = ChapterFinalizer::new(&registry, &sink);
        let model = MockLanguageModel::new(vec![""]);
        let request = EnrichChapterRequest {
            chapter_number: Some(1),
            chapter_text: "原文",
            target_word_count: 500,
        };

        let result = finalizer.enrich_chapter_text(&model, &request).unwrap();
        assert_eq!(result, "原文");
    }

    #[test]
    fn finalize_preserves_existing_files_on_empty_response() {
        let registry = PromptRegistry::new().unwrap();
        let sink = NullLogSink;
        let finalizer = ChapterFinalizer::new(&registry, &sink);
        let model = MockLanguageModel::new(vec![""]);

        let temp_dir = tempdir().unwrap();
        let chapters_dir = temp_dir.path().join(CHAPTERS_DIR_NAME);
        fs::create_dir_all(&chapters_dir).unwrap();
        let chapter_path = chapters_dir.join("chapter_1.txt");
        fs::write(&chapter_path, "示例章节").unwrap();

        let summary_path = temp_dir.path().join(GLOBAL_SUMMARY_FILE_NAME);
        fs::write(&summary_path, "旧摘要").unwrap();
        let character_state_path = temp_dir.path().join(CHARACTER_STATE_FILE_NAME);
        fs::write(&character_state_path, "旧角色状态").unwrap();

        let request = FinalizeChapterRequest {
            output_dir: temp_dir.path().to_path_buf(),
            chapter_number: 1,
        };

        let result = finalizer.finalize_chapter(&model, None, &request).unwrap();

        assert_eq!(result.summary_text, "旧摘要");
        assert_eq!(result.character_state_text, "旧角色状态");
    }
}
