use crate::architecture::{LanguageModel, LanguageModelError, ARCHITECTURE_FILE_NAME};
use crate::logging::{LogLevel, LogRecord, LogSink};
use crate::prompts::{PromptError, PromptRegistry};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::min;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use thiserror::Error;

pub const BLUEPRINT_FILE_NAME: &str = "Novel_directory.txt";
const BLUEPRINT_LIMIT: usize = 100;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChapterBlueprintEntry {
    pub chapter_number: u32,
    pub chapter_title: String,
    pub chapter_role: String,
    pub chapter_purpose: String,
    pub suspense_level: String,
    pub foreshadowing: String,
    pub plot_twist_level: String,
    pub chapter_summary: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChapterBlueprint {
    raw_text: String,
    chapters: Vec<ChapterBlueprintEntry>,
}

impl ChapterBlueprint {
    pub fn from_text(raw_text: String) -> Self {
        let trimmed = raw_text.trim().to_string();
        let chapters = parse_chapter_blueprint(&trimmed);
        Self {
            raw_text: trimmed,
            chapters,
        }
    }

    pub fn raw_text(&self) -> &str {
        &self.raw_text
    }

    pub fn chapters(&self) -> &[ChapterBlueprintEntry] {
        &self.chapters
    }

    pub fn chapter(&self, number: u32) -> Option<&ChapterBlueprintEntry> {
        self.chapters
            .iter()
            .find(|entry| entry.chapter_number == number)
    }

    pub fn len(&self) -> usize {
        self.chapters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    pub fn max_chapter_number(&self) -> Option<u32> {
        self.chapters.iter().map(|entry| entry.chapter_number).max()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChapterBlueprintRequest {
    pub number_of_chapters: u32,
    pub user_guidance: String,
    pub max_tokens: u32,
}

impl ChapterBlueprintRequest {
    pub fn new(number_of_chapters: u32, user_guidance: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            number_of_chapters,
            user_guidance: user_guidance.into(),
            max_tokens,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlueprintPromptKind {
    Initial,
    Chunk { start: u32, end: u32 },
}

impl BlueprintPromptKind {
    fn prompt_key(&self) -> &'static str {
        match self {
            Self::Initial => "chapter_blueprint",
            Self::Chunk { .. } => "chunked_chapter_blueprint",
        }
    }
}

impl fmt::Display for BlueprintPromptKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initial => write!(f, "章节蓝图-单次生成"),
            Self::Chunk { start, end } => write!(f, "章节蓝图-分块({start}-{end})"),
        }
    }
}

#[derive(Debug, Error)]
pub enum BlueprintError {
    #[error("failed to prepare output directory `{path}`: {source}")]
    CreateDir { path: PathBuf, source: io::Error },
    #[error("failed to read architecture file `{path}`: {source}")]
    ReadArchitecture { path: PathBuf, source: io::Error },
    #[error("architecture file `{path}` is empty")]
    EmptyArchitecture { path: PathBuf },
    #[error("failed to read blueprint file `{path}`: {source}")]
    ReadBlueprint { path: PathBuf, source: io::Error },
    #[error("failed to write blueprint file `{path}`: {source}")]
    WriteBlueprint { path: PathBuf, source: io::Error },
    #[error("failed to render {kind} prompt: {source}")]
    Prompt {
        kind: BlueprintPromptKind,
        #[source]
        source: PromptError,
    },
    #[error("language model invocation failed for {kind}: {source}")]
    Model {
        kind: BlueprintPromptKind,
        #[source]
        source: LanguageModelError,
    },
    #[error("{kind} response was empty")]
    EmptyResponse { kind: BlueprintPromptKind },
}

pub struct ChapterBlueprintService<'a> {
    prompts: &'a PromptRegistry,
    sink: &'a dyn LogSink,
    max_retries: usize,
    cache: RefCell<Option<CachedBlueprint>>,
}

#[derive(Clone)]
struct CachedBlueprint {
    output_dir: PathBuf,
    blueprint: ChapterBlueprint,
}

impl<'a> ChapterBlueprintService<'a> {
    pub fn new(prompts: &'a PromptRegistry, sink: &'a dyn LogSink) -> Self {
        Self {
            prompts,
            sink,
            max_retries: 3,
            cache: RefCell::new(None),
        }
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries.max(1);
        self
    }

    pub fn generate<M: LanguageModel>(
        &self,
        model: &M,
        output_dir: impl AsRef<Path>,
        request: &ChapterBlueprintRequest,
    ) -> Result<ChapterBlueprint, BlueprintError> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir).map_err(|source| BlueprintError::CreateDir {
            path: output_dir.to_path_buf(),
            source,
        })?;

        let architecture = self.read_architecture_text(output_dir)?;
        let blueprint_path = output_dir.join(BLUEPRINT_FILE_NAME);
        let existing_content = match fs::read_to_string(&blueprint_path) {
            Ok(text) => text,
            Err(err) if err.kind() == io::ErrorKind::NotFound => String::new(),
            Err(err) => {
                return Err(BlueprintError::ReadBlueprint {
                    path: blueprint_path.clone(),
                    source: err,
                })
            }
        };

        let mut final_text = existing_content.trim().to_string();
        let chunk_size = compute_chunk_size(request.number_of_chapters, request.max_tokens);

        self.log(
            LogLevel::Info,
            format!(
                "章节总数 = {}，计算出的 chunk_size = {}。",
                request.number_of_chapters, chunk_size
            ),
        );

        let mut wrote_file = false;

        if final_text.is_empty() {
            if chunk_size >= request.number_of_chapters as usize {
                let prompt = self
                    .prompts
                    .format_with(
                        BlueprintPromptKind::Initial.prompt_key(),
                        [
                            ("novel_architecture", architecture.clone()),
                            ("number_of_chapters", request.number_of_chapters.to_string()),
                            ("user_guidance", request.user_guidance.trim().to_string()),
                        ],
                    )
                    .map_err(|source| BlueprintError::Prompt {
                        kind: BlueprintPromptKind::Initial,
                        source,
                    })?;

                let response =
                    self.invoke_with_cleaning(model, BlueprintPromptKind::Initial, &prompt)?;
                if response.trim().is_empty() {
                    return Err(BlueprintError::EmptyResponse {
                        kind: BlueprintPromptKind::Initial,
                    });
                }

                final_text = response.trim().to_string();
                self.write_blueprint_file(&blueprint_path, &final_text)?;
                wrote_file = true;
                self.log(
                    LogLevel::Info,
                    "Novel_directory.txt (chapter blueprint) has been generated successfully (single-shot).",
                );
            } else {
                self.log(LogLevel::Info, "将以分块模式从头生成章节蓝图。");
                let mut current_start = 1u32;
                while current_start <= request.number_of_chapters {
                    let current_end = min(
                        current_start + chunk_size as u32 - 1,
                        request.number_of_chapters,
                    );
                    let limited = limit_chapter_blueprint(&final_text, BLUEPRINT_LIMIT);
                    let prompt = self
                        .prompts
                        .format_with(
                            BlueprintPromptKind::Chunk {
                                start: current_start,
                                end: current_end,
                            }
                            .prompt_key(),
                            [
                                ("novel_architecture", architecture.clone()),
                                ("chapter_list", limited),
                                ("number_of_chapters", request.number_of_chapters.to_string()),
                                ("n", current_start.to_string()),
                                ("m", current_end.to_string()),
                                ("user_guidance", request.user_guidance.trim().to_string()),
                            ],
                        )
                        .map_err(|source| BlueprintError::Prompt {
                            kind: BlueprintPromptKind::Chunk {
                                start: current_start,
                                end: current_end,
                            },
                            source,
                        })?;

                    self.log(
                        LogLevel::Info,
                        format!("正在生成第{}-{}章的目录...", current_start, current_end),
                    );

                    let response = self.invoke_with_cleaning(
                        model,
                        BlueprintPromptKind::Chunk {
                            start: current_start,
                            end: current_end,
                        },
                        &prompt,
                    )?;

                    let trimmed = response.trim();
                    if trimmed.is_empty() {
                        self.write_blueprint_file(&blueprint_path, &final_text)?;
                        return Err(BlueprintError::EmptyResponse {
                            kind: BlueprintPromptKind::Chunk {
                                start: current_start,
                                end: current_end,
                            },
                        });
                    }

                    if final_text.is_empty() {
                        final_text = trimmed.to_string();
                    } else {
                        final_text.push_str("\n\n");
                        final_text.push_str(trimmed);
                    }

                    self.write_blueprint_file(&blueprint_path, &final_text)?;
                    wrote_file = true;
                    current_start = current_end + 1;
                }

                self.log(
                    LogLevel::Info,
                    "Novel_directory.txt (chapter blueprint) has been generated successfully (chunked).",
                );
            }
        } else {
            self.log(
                LogLevel::Info,
                "检测到已有章节蓝图内容，将从该进度继续分块生成。",
            );
            let max_existing = find_max_chapter_number(&final_text).unwrap_or(0);
            self.log(
                LogLevel::Info,
                format!("现有蓝图已生成至第{}章。", max_existing),
            );
            let mut current_start = max_existing.saturating_add(1);

            while current_start <= request.number_of_chapters {
                let current_end = min(
                    current_start + chunk_size as u32 - 1,
                    request.number_of_chapters,
                );
                let limited = limit_chapter_blueprint(&final_text, BLUEPRINT_LIMIT);
                let prompt = self
                    .prompts
                    .format_with(
                        BlueprintPromptKind::Chunk {
                            start: current_start,
                            end: current_end,
                        }
                        .prompt_key(),
                        [
                            ("novel_architecture", architecture.clone()),
                            ("chapter_list", limited),
                            ("number_of_chapters", request.number_of_chapters.to_string()),
                            ("n", current_start.to_string()),
                            ("m", current_end.to_string()),
                            ("user_guidance", request.user_guidance.trim().to_string()),
                        ],
                    )
                    .map_err(|source| BlueprintError::Prompt {
                        kind: BlueprintPromptKind::Chunk {
                            start: current_start,
                            end: current_end,
                        },
                        source,
                    })?;

                self.log(
                    LogLevel::Info,
                    format!("正在继续生成第{}-{}章的目录...", current_start, current_end),
                );

                let response = self.invoke_with_cleaning(
                    model,
                    BlueprintPromptKind::Chunk {
                        start: current_start,
                        end: current_end,
                    },
                    &prompt,
                )?;

                let trimmed = response.trim();
                if trimmed.is_empty() {
                    self.write_blueprint_file(&blueprint_path, &final_text)?;
                    return Err(BlueprintError::EmptyResponse {
                        kind: BlueprintPromptKind::Chunk {
                            start: current_start,
                            end: current_end,
                        },
                    });
                }

                if final_text.is_empty() {
                    final_text = trimmed.to_string();
                } else {
                    final_text.push_str("\n\n");
                    final_text.push_str(trimmed);
                }

                self.write_blueprint_file(&blueprint_path, &final_text)?;
                wrote_file = true;
                current_start = current_end + 1;
            }

            if wrote_file {
                self.log(LogLevel::Info, "所有章节蓝图已补全（续跑模式）。");
            }
        }

        if !wrote_file {
            self.write_blueprint_file(&blueprint_path, &final_text)?;
        }

        let blueprint = ChapterBlueprint::from_text(final_text);
        self.update_cache(output_dir, &blueprint);
        Ok(blueprint)
    }

    pub fn load(
        &self,
        output_dir: impl AsRef<Path>,
    ) -> Result<Option<ChapterBlueprint>, BlueprintError> {
        let output_dir = output_dir.as_ref();
        if let Some(cached) = self.cache.borrow().as_ref() {
            if cached.output_dir == output_dir {
                return Ok(Some(cached.blueprint.clone()));
            }
        }

        let blueprint_path = output_dir.join(BLUEPRINT_FILE_NAME);
        let content = match fs::read_to_string(&blueprint_path) {
            Ok(text) => text,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(BlueprintError::ReadBlueprint {
                    path: blueprint_path,
                    source: err,
                })
            }
        };

        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let blueprint = ChapterBlueprint::from_text(trimmed.to_string());
        self.update_cache(output_dir, &blueprint);
        Ok(Some(blueprint))
    }

    fn update_cache(&self, output_dir: &Path, blueprint: &ChapterBlueprint) {
        self.cache.replace(Some(CachedBlueprint {
            output_dir: output_dir.to_path_buf(),
            blueprint: blueprint.clone(),
        }));
    }

    fn read_architecture_text(&self, output_dir: &Path) -> Result<String, BlueprintError> {
        let path = output_dir.join(ARCHITECTURE_FILE_NAME);
        let content =
            fs::read_to_string(&path).map_err(|source| BlueprintError::ReadArchitecture {
                path: path.clone(),
                source,
            })?;
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return Err(BlueprintError::EmptyArchitecture { path });
        }
        Ok(trimmed)
    }

    fn write_blueprint_file(&self, path: &Path, content: &str) -> Result<(), BlueprintError> {
        fs::write(path, content.trim()).map_err(|source| BlueprintError::WriteBlueprint {
            path: path.to_path_buf(),
            source,
        })
    }

    fn invoke_with_cleaning<M: LanguageModel>(
        &self,
        model: &M,
        kind: BlueprintPromptKind,
        prompt: &str,
    ) -> Result<String, BlueprintError> {
        for attempt in 1..=self.max_retries {
            self.log(
                LogLevel::Info,
                format!(
                    "发送到 LLM 的提示词（{}｜第{}次尝试）：\n{}",
                    kind, attempt, prompt
                ),
            );

            match model.invoke(prompt) {
                Ok(response) => {
                    self.log(
                        LogLevel::Info,
                        format!(
                            "LLM 返回的内容（{}｜第{}次尝试）：\n{}",
                            kind, attempt, response
                        ),
                    );
                    let cleaned = response.replace("```", "").trim().to_string();
                    if !cleaned.is_empty() {
                        return Ok(cleaned);
                    }

                    self.log(
                        LogLevel::Warn,
                        format!("LLM 返回空响应，准备重试（{}｜第{}次尝试）", kind, attempt),
                    );
                }
                Err(err) => {
                    let message = err.to_string();
                    self.log(
                        LogLevel::Warn,
                        format!("LLM 调用失败（{}｜第{}次尝试）：{}", kind, attempt, message),
                    );
                    if attempt == self.max_retries {
                        return Err(BlueprintError::Model { kind, source: err });
                    }
                }
            }
        }

        Ok(String::new())
    }

    fn log(&self, level: LogLevel, message: impl Into<String>) {
        self.sink.log(LogRecord::new(level, message.into()));
    }
}

pub fn compute_chunk_size(number_of_chapters: u32, max_tokens: u32) -> usize {
    if number_of_chapters == 0 {
        return 0;
    }

    let ratio = (max_tokens as f64) / 100.0;
    let ratio_rounded_to_10 = ((ratio / 10.0).floor() as i32) * 10;
    let mut chunk_size = ratio_rounded_to_10 - 10;
    if chunk_size < 1 {
        chunk_size = 1;
    }
    let chunk_size = chunk_size.min(number_of_chapters as i32);
    chunk_size.max(0) as usize
}

pub fn limit_chapter_blueprint(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut boundaries: Vec<usize> = finder_regex()
        .find_iter(trimmed)
        .map(|m| m.start())
        .collect();

    if boundaries.is_empty() {
        return trimmed.to_string();
    }

    boundaries.push(trimmed.len());
    let segments: Vec<String> = boundaries
        .windows(2)
        .map(|window| {
            let start = window[0];
            let end = window[1];
            trimmed[start..end].trim().to_string()
        })
        .collect();

    if segments.len() <= limit {
        return trimmed.to_string();
    }

    segments
        .into_iter()
        .rev()
        .take(limit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn parse_chapter_blueprint(text: &str) -> Vec<ChapterBlueprintEntry> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    for chunk in block_split_regex().split(trimmed) {
        let block = chunk.trim();
        if block.is_empty() {
            continue;
        }

        let mut lines: Vec<&str> = block.lines().map(|line| line.trim()).collect();
        lines.retain(|line| !line.is_empty());
        if lines.is_empty() {
            continue;
        }

        let Some(header) = header_regex().captures(lines[0]) else {
            continue;
        };

        let chapter_number: u32 = header
            .get(1)
            .and_then(|m| m.as_str().trim().parse().ok())
            .unwrap_or(0);
        let chapter_title = header
            .get(2)
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();

        let mut entry = ChapterBlueprintEntry {
            chapter_number,
            chapter_title,
            chapter_role: String::new(),
            chapter_purpose: String::new(),
            suspense_level: String::new(),
            foreshadowing: String::new(),
            plot_twist_level: String::new(),
            chapter_summary: String::new(),
        };

        for line in &lines[1..] {
            if let Some(caps) = role_regex().captures(line) {
                entry.chapter_role = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                continue;
            }
            if let Some(caps) = purpose_regex().captures(line) {
                entry.chapter_purpose = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                continue;
            }
            if let Some(caps) = suspense_regex().captures(line) {
                entry.suspense_level = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                continue;
            }
            if let Some(caps) = foreshadow_regex().captures(line) {
                entry.foreshadowing = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                continue;
            }
            if let Some(caps) = twist_regex().captures(line) {
                entry.plot_twist_level = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                continue;
            }
            if let Some(caps) = summary_regex().captures(line) {
                entry.chapter_summary = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                continue;
            }
        }

        entries.push(entry);
    }

    entries.sort_by_key(|entry| entry.chapter_number);
    entries
}

fn find_max_chapter_number(text: &str) -> Option<u32> {
    finder_regex()
        .captures_iter(text)
        .filter_map(|caps| caps.get(1))
        .filter_map(|m| m.as_str().trim().parse::<u32>().ok())
        .max()
}

fn block_split_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\n\s*\n").expect("invalid blueprint splitter"))
}

fn header_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^第\s*(\d+)\s*章\s*-\s*\[?(.*?)\]?$").expect("invalid chapter header regex")
    })
}

fn role_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^本章定位[：:]\s*\[?(.*?)\]?$").expect("invalid role regex"))
}

fn purpose_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^核心作用[：:]\s*\[?(.*?)\]?$").expect("invalid purpose regex")
    })
}

fn suspense_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^悬念密度[：:]\s*\[?(.*?)\]?$").expect("invalid suspense regex")
    })
}

fn foreshadow_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^伏笔操作[：:]\s*\[?(.*?)\]?$").expect("invalid foreshadow regex")
    })
}

fn twist_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^认知颠覆[：:]\s*\[?(.*?)\]?$").expect("invalid twist regex"))
}

fn summary_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^本章简述[：:]\s*\[?(.*?)\]?$").expect("invalid summary regex")
    })
}

fn finder_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"第\s*(\d+)\s*章").expect("invalid chapter finder regex"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::VecLogSink;
    use crate::prompts::PromptRegistry;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use tempfile::TempDir;

    struct MockLanguageModel {
        responses: Mutex<VecDeque<String>>,
        prompts: Mutex<Vec<String>>,
    }

    impl MockLanguageModel {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(|s| s.to_string()).collect()),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts.lock().unwrap().clone()
        }
    }

    impl LanguageModel for MockLanguageModel {
        fn invoke(&self, prompt: &str) -> Result<String, LanguageModelError> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            self.responses.lock().unwrap().pop_front().ok_or_else(|| {
                LanguageModelError::new(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "no more mock responses",
                ))
            })
        }
    }

    #[test]
    fn parse_multiple_chapters() {
        let text = r#"
第1章 - [紫极光下的预兆]
本章定位：[角色线]
核心作用：[铺垫世界观]
悬念密度：[渐进]
伏笔操作：[埋设-遗迹闪光]
认知颠覆：[★☆☆☆☆]
本章简述：[主人公第一次见到紫极光，内心充满未知]

第2章 - [风暴前的协议]
本章定位：政治冲突
核心作用：推进矛盾
悬念密度：[紧凑]
伏笔操作：[强化-盟约]
认知颠覆：★★☆☆☆
本章简述：主人公与反抗军签订协议，暗藏危机
"#;

        let chapters = parse_chapter_blueprint(text);
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].chapter_number, 1);
        assert_eq!(chapters[0].chapter_title, "紫极光下的预兆");
        assert_eq!(chapters[0].chapter_role, "角色线");
        assert_eq!(chapters[1].chapter_number, 2);
        assert_eq!(chapters[1].chapter_title, "风暴前的协议");
        assert_eq!(chapters[1].chapter_purpose, "推进矛盾");
        assert_eq!(chapters[1].plot_twist_level, "★★☆☆☆");
    }

    #[test]
    fn resume_generation_appends_new_chapters() {
        let temp = TempDir::new().expect("temp dir");
        let output_dir = temp.path();
        fs::write(output_dir.join(ARCHITECTURE_FILE_NAME), "核心设定").unwrap();

        let existing = "第1章 - [开端]\n本章定位：[引入]\n核心作用：[铺垫]\n悬念密度：[渐进]\n伏笔操作：[埋设]\n认知颠覆：[★☆☆☆☆]\n本章简述：[介绍故事背景]\n\n第2章 - [冲突]\n本章定位：[冲突]\n核心作用：[推进]\n悬念密度：[紧凑]\n伏笔操作：[强化]\n认知颠覆：[★★☆☆☆]\n本章简述：[主要冲突展开]";
        fs::write(output_dir.join(BLUEPRINT_FILE_NAME), existing).unwrap();

        let prompts = PromptRegistry::new().expect("registry");
        let sink = VecLogSink::new();
        let service = ChapterBlueprintService::new(&prompts, &sink).with_max_retries(1);

        let chunk3 = "第3章 - [转折]\n本章定位：[角色]\n核心作用：[转折]\n悬念密度：[紧凑]\n伏笔操作：[埋设]\n认知颠覆：[★★★☆☆]\n本章简述：[第三章总结]";
        let chunk4 = "第4章 - [终章]\n本章定位：[高潮]\n核心作用：[收束]\n悬念密度：[爆发]\n伏笔操作：[回收]\n认知颠覆：[★★★★★]\n本章简述：[第四章总结]";
        let mock = MockLanguageModel::new(vec![chunk3, chunk4]);

        let request = ChapterBlueprintRequest::new(4, "测试", 1000);
        let result = service
            .generate(&mock, output_dir, &request)
            .expect("generation should succeed");

        assert_eq!(result.chapters().len(), 4);
        assert_eq!(result.chapter(3).unwrap().chapter_title, "转折");
        assert_eq!(result.chapter(4).unwrap().chapter_summary, "第四章总结");

        let expected = [existing.trim(), chunk3.trim(), chunk4.trim()].join("\n\n");
        let stored = fs::read_to_string(output_dir.join(BLUEPRINT_FILE_NAME)).unwrap();
        assert_eq!(stored, expected);

        let prompts_used = mock.prompts();
        assert!(prompts_used
            .iter()
            .any(|prompt| prompt.contains("第1章 - [开端]")));
        assert!(prompts_used
            .iter()
            .any(|prompt| prompt.contains("第3章 - [转折]")));
    }
}
