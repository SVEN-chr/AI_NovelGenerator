use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::config::PromptConfig;

const BUILT_IN_PROMPTS: &str = include_str!("../../prompts/default.toml");

pub type PromptArguments = HashMap<String, String>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptSource {
    BuiltIn,
    File(PathBuf),
}

impl PromptSource {
    pub fn is_builtin(&self) -> bool {
        matches!(self, Self::BuiltIn)
    }

    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::BuiltIn => None,
            Self::File(path) => Some(path.as_path()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PromptMetadata {
    description: Option<String>,
    source: PromptSource,
}

impl PromptMetadata {
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn source(&self) -> &PromptSource {
        &self.source
    }
}

#[derive(Clone, Debug)]
pub struct PromptTemplate {
    key: String,
    template: String,
    segments: Vec<TemplateSegment>,
    placeholders: BTreeSet<String>,
    required: BTreeSet<String>,
    metadata: PromptMetadata,
}

impl PromptTemplate {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn template(&self) -> &str {
        &self.template
    }

    pub fn placeholders(&self) -> impl Iterator<Item = &str> {
        self.placeholders.iter().map(|s| s.as_str())
    }

    pub fn required_arguments(&self) -> impl Iterator<Item = &str> {
        self.required.iter().map(|s| s.as_str())
    }

    pub fn metadata(&self) -> &PromptMetadata {
        &self.metadata
    }

    pub fn render(&self, arguments: &PromptArguments) -> Result<String, PromptError> {
        for required in &self.required {
            if !arguments.contains_key(required) {
                return Err(PromptError::MissingArgument {
                    key: self.key.clone(),
                    argument: required.clone(),
                });
            }
        }

        let mut output = String::with_capacity(self.template.len());
        for segment in &self.segments {
            match segment {
                TemplateSegment::Literal(text) => output.push_str(text),
                TemplateSegment::Placeholder(name) => {
                    if let Some(value) = arguments.get(name) {
                        output.push_str(value);
                    }
                }
            }
        }

        Ok(output)
    }

    pub fn render_with<I, K, V>(&self, arguments: I) -> Result<String, PromptError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut map = PromptArguments::new();
        for (key, value) in arguments {
            map.insert(key.into(), value.into());
        }
        self.render(&map)
    }

    fn from_raw(key: String, raw: RawPrompt, source: PromptSource) -> Result<Self, PromptError> {
        let (segments, placeholders) = parse_template(&raw.template);
        let required = if raw.required.is_empty() {
            placeholders.clone()
        } else {
            let mut set = BTreeSet::new();
            for argument in raw.required {
                let trimmed = argument.trim().to_string();
                if !placeholders.contains(&trimmed) {
                    return Err(PromptError::InvalidRequired {
                        key: key.clone(),
                        argument: trimmed,
                    });
                }
                set.insert(trimmed);
            }
            set
        };

        Ok(Self {
            key,
            template: raw.template,
            segments,
            placeholders,
            required,
            metadata: PromptMetadata {
                description: raw.description,
                source,
            },
        })
    }
}

#[derive(Debug, Error)]
pub enum PromptError {
    #[error("prompt `{0}` not found")]
    NotFound(String),
    #[error("missing argument `{argument}` when rendering prompt `{key}`")]
    MissingArgument { key: String, argument: String },
    #[error("failed to read prompt file `{path}`: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse built-in prompt definitions: {0}")]
    ParseBuiltIn(toml::de::Error),
    #[error("failed to parse prompt file `{path}` as TOML: {source}")]
    ParseToml {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("failed to parse prompt file `{path}` as YAML: {source}")]
    ParseYaml {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    #[error(
        "required key `{argument}` declared for prompt `{key}` but no matching placeholder was found"
    )]
    InvalidRequired { key: String, argument: String },
}

#[derive(Debug)]
pub struct PromptRegistry {
    prompts: BTreeMap<String, PromptTemplate>,
    directories: Vec<PathBuf>,
    hot_reload: bool,
}

impl PromptRegistry {
    pub fn new() -> Result<Self, PromptError> {
        Self::from_prompt_config(&PromptConfig::default())
    }

    pub fn from_prompt_config(config: &PromptConfig) -> Result<Self, PromptError> {
        Self::with_options(config.custom_directories.clone(), config.enable_hot_reload)
    }

    pub fn with_custom_directories<P: AsRef<Path>>(directories: &[P]) -> Result<Self, PromptError> {
        let dirs = directories
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        Self::with_options(dirs, false)
    }

    pub fn hot_reload_enabled(&self) -> bool {
        self.hot_reload
    }

    pub fn custom_directories(&self) -> &[PathBuf] {
        &self.directories
    }

    pub fn reload(&mut self) -> Result<(), PromptError> {
        self.prompts = Self::build_prompts(&self.directories)?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&PromptTemplate> {
        self.prompts.get(key)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.prompts.contains_key(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.prompts.keys().map(|k| k.as_str())
    }

    pub fn format(&self, key: &str, args: &PromptArguments) -> Result<String, PromptError> {
        let template = self
            .get(key)
            .ok_or_else(|| PromptError::NotFound(key.to_string()))?;
        template.render(args)
    }

    pub fn format_with<I, K, V>(&self, key: &str, arguments: I) -> Result<String, PromptError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let template = self
            .get(key)
            .ok_or_else(|| PromptError::NotFound(key.to_string()))?;
        template.render_with(arguments)
    }

    fn with_options(directories: Vec<PathBuf>, hot_reload: bool) -> Result<Self, PromptError> {
        let mut registry = Self {
            prompts: BTreeMap::new(),
            directories,
            hot_reload,
        };
        registry.reload()?;
        Ok(registry)
    }

    fn build_prompts(
        directories: &[PathBuf],
    ) -> Result<BTreeMap<String, PromptTemplate>, PromptError> {
        let mut prompts = BTreeMap::new();

        let built_in = parse_document(BUILT_IN_PROMPTS, PromptSource::BuiltIn)?;
        for template in built_in {
            prompts.insert(template.key().to_string(), template);
        }

        for dir in directories {
            load_directory(dir, &mut prompts)?;
        }

        Ok(prompts)
    }
}

fn load_directory(
    dir: &Path,
    prompts: &mut BTreeMap<String, PromptTemplate>,
) -> Result<(), PromptError> {
    if !dir.exists() {
        return Ok(());
    }
    if !dir.is_dir() {
        return Ok(());
    }

    let mut files = Vec::new();
    let read_dir = fs::read_dir(dir).map_err(|source| PromptError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in read_dir {
        let entry = entry.map_err(|source| PromptError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        match entry.file_type() {
            Ok(file_type) if file_type.is_file() => files.push(path),
            Ok(_) => {}
            Err(source) => {
                return Err(PromptError::Io {
                    path: path.clone(),
                    source,
                })
            }
        }
    }

    files.sort();

    for path in files {
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        match ext.to_ascii_lowercase().as_str() {
            "toml" => {
                let contents = fs::read_to_string(&path).map_err(|source| PromptError::Io {
                    path: path.clone(),
                    source,
                })?;
                let templates = parse_document(&contents, PromptSource::File(path.clone()))
                    .map_err(|err| match err {
                        PromptError::ParseBuiltIn(source) => PromptError::ParseToml {
                            path: path.clone(),
                            source,
                        },
                        other => other,
                    })?;
                for template in templates {
                    prompts.insert(template.key().to_string(), template);
                }
            }
            "yaml" | "yml" => {
                let contents = fs::read_to_string(&path).map_err(|source| PromptError::Io {
                    path: path.clone(),
                    source,
                })?;
                let document: PromptDocument =
                    serde_yaml::from_str(&contents).map_err(|source| PromptError::ParseYaml {
                        path: path.clone(),
                        source,
                    })?;
                for (key, raw) in document.prompts {
                    let template = PromptTemplate::from_raw(
                        key.clone(),
                        raw,
                        PromptSource::File(path.clone()),
                    )?;
                    prompts.insert(key, template);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn parse_document(source: &str, origin: PromptSource) -> Result<Vec<PromptTemplate>, PromptError> {
    let document: PromptDocument = toml::from_str(source).map_err(PromptError::ParseBuiltIn)?;
    let mut templates = Vec::new();
    for (key, raw) in document.prompts {
        let template = PromptTemplate::from_raw(key.clone(), raw, origin.clone())?;
        templates.push(template);
    }
    Ok(templates)
}

#[derive(Debug, Deserialize)]
struct PromptDocument {
    #[serde(default)]
    prompts: BTreeMap<String, RawPrompt>,
}

#[derive(Debug, Deserialize)]
struct RawPrompt {
    #[serde(alias = "text")]
    template: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    required: Vec<String>,
}

#[derive(Clone, Debug)]
enum TemplateSegment {
    Literal(String),
    Placeholder(String),
}

fn parse_template(template: &str) -> (Vec<TemplateSegment>, BTreeSet<String>) {
    let mut segments = Vec::new();
    let mut placeholders = BTreeSet::new();
    let mut buffer = String::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if matches!(chars.peek(), Some('{')) {
                    chars.next();
                    buffer.push('{');
                    continue;
                }

                if !buffer.is_empty() {
                    segments.push(TemplateSegment::Literal(std::mem::take(&mut buffer)));
                }

                let mut placeholder = String::new();
                let mut closed = false;
                while let Some(next) = chars.next() {
                    if next == '}' {
                        closed = true;
                        break;
                    } else {
                        placeholder.push(next);
                    }
                }

                if closed {
                    let trimmed = placeholder.trim();
                    if trimmed.is_empty() {
                        segments.push(TemplateSegment::Literal("{}".to_string()));
                    } else {
                        let key = trimmed.to_string();
                        placeholders.insert(key.clone());
                        segments.push(TemplateSegment::Placeholder(key));
                    }
                } else {
                    buffer.push('{');
                    buffer.push_str(&placeholder);
                }
            }
            '}' => {
                if matches!(chars.peek(), Some('}')) {
                    chars.next();
                    buffer.push('}');
                } else {
                    buffer.push('}');
                }
            }
            _ => buffer.push(ch),
        }
    }

    if !buffer.is_empty() {
        segments.push(TemplateSegment::Literal(buffer));
    }

    (segments, placeholders)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn sample_args() -> PromptArguments {
        let mut args = PromptArguments::new();
        args.insert("combined_text".into(), "第一章...".into());
        args.insert("novel_number".into(), "4".into());
        args.insert("chapter_title".into(), "迷雾重构".into());
        args.insert("chapter_role".into(), "推进主线".into());
        args.insert("chapter_purpose".into(), "揭示真相".into());
        args.insert("suspense_level".into(), "高".into());
        args.insert("foreshadowing".into(), "埋设线索".into());
        args.insert("plot_twist_level".into(), "★★☆☆☆".into());
        args.insert("chapter_summary".into(), "团队发现隐藏实验室".into());
        args.insert("next_chapter_number".into(), "5".into());
        args.insert("next_chapter_title".into(), "双重盲区".into());
        args.insert("next_chapter_role".into(), "冲突升级".into());
        args.insert("next_chapter_purpose".into(), "制造悬念".into());
        args.insert("next_chapter_suspense_level".into(), "极高".into());
        args.insert("next_chapter_foreshadowing".into(), "强化伏笔".into());
        args.insert("next_chapter_plot_twist_level".into(), "★★★☆☆".into());
        args.insert("next_chapter_summary".into(), "敌人掌控局面".into());
        args
    }

    #[test]
    fn renders_default_prompt() {
        let registry = PromptRegistry::new().expect("registry");
        let output = registry
            .format("summarize_recent_chapters", &sample_args())
            .expect("rendered");
        assert!(output.contains("当前章节摘要"));
        assert!(output.contains("前三章内容"));
    }

    #[test]
    fn missing_argument_fails() {
        let registry = PromptRegistry::new().expect("registry");
        let template = registry.get("core_seed").expect("core_seed available");
        let args = PromptArguments::from([("topic".into(), "AI科幻".into())]);
        let error = template.render(&args).expect_err("missing args");
        match error {
            PromptError::MissingArgument { argument, .. } => {
                assert_eq!(argument, "genre");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn custom_directory_overrides() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("custom.toml");
        fs::write(&path, "[prompts.core_seed]\ntemplate = \"定制 {topic}\"\n").unwrap();

        let registry = PromptRegistry::with_custom_directories(&[dir.path()]).unwrap();
        let output = registry
            .format(
                "core_seed",
                &PromptArguments::from([("topic".into(), "悬疑".into())]),
            )
            .unwrap();
        assert!(output.contains("定制 悬疑"));
    }

    #[test]
    fn reload_reflects_changes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("custom.toml");
        fs::write(
            &path,
            "[prompts.summary]\ntemplate = \"初始 {chapter_text}\"\n",
        )
        .unwrap();

        let mut registry = PromptRegistry::with_options(vec![dir.path().into()], true).unwrap();
        let first = registry
            .format(
                "summary",
                &PromptArguments::from([
                    ("chapter_text".into(), "第一章".into()),
                    ("global_summary".into(), "".into()),
                ]),
            )
            .unwrap();
        assert_eq!(first, "初始 第一章");

        fs::write(
            &path,
            "[prompts.summary]\ntemplate = \"更新 {chapter_text}\"\n",
        )
        .unwrap();

        registry.reload().unwrap();
        let second = registry
            .format(
                "summary",
                &PromptArguments::from([
                    ("chapter_text".into(), "第二章".into()),
                    ("global_summary".into(), "".into()),
                ]),
            )
            .unwrap();
        assert_eq!(second, "更新 第二章");
    }
}
