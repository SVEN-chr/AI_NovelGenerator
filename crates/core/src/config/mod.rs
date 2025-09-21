use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_timeout() -> u64 {
    600
}

fn default_embedding_retrieval_k() -> u32 {
    4
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to access config: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Parse(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LlmConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub interface_format: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: String::new(),
            interface_format: String::new(),
            model_name: String::new(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            timeout: default_timeout(),
        }
    }
}

impl LlmConfig {
    pub fn is_meaningful(&self) -> bool {
        !(self.api_key.is_empty()
            && self.base_url.is_empty()
            && self.interface_format.is_empty()
            && self.model_name.is_empty())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub interface_format: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default = "default_embedding_retrieval_k")]
    pub retrieval_k: u32,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: String::new(),
            interface_format: String::new(),
            model_name: String::new(),
            retrieval_k: default_embedding_retrieval_k(),
        }
    }
}

impl EmbeddingConfig {
    pub fn is_meaningful(&self) -> bool {
        !(self.api_key.is_empty()
            && self.base_url.is_empty()
            && self.interface_format.is_empty()
            && self.model_name.is_empty())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NovelConfig {
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub genre: String,
    #[serde(default)]
    pub num_chapters: u32,
    #[serde(default)]
    pub word_number: u32,
    #[serde(default)]
    pub filepath: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RecentUsage {
    #[serde(default)]
    pub last_llm_interface: Option<String>,
    #[serde(default)]
    pub last_embedding_interface: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Config {
    #[serde(default)]
    pub llm_profiles: BTreeMap<String, LlmConfig>,
    #[serde(default)]
    pub embedding_profiles: BTreeMap<String, EmbeddingConfig>,
    #[serde(default)]
    pub novel: NovelConfig,
    #[serde(default)]
    pub recent: RecentUsage,
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_llm_profile(&self, name: &str) -> Option<&LlmConfig> {
        self.llm_profiles.get(name)
    }

    pub fn upsert_llm_profile<S: Into<String>>(&mut self, name: S, profile: LlmConfig) {
        self.llm_profiles.insert(name.into(), profile);
    }

    pub fn remove_llm_profile(&mut self, name: &str) -> Option<LlmConfig> {
        self.llm_profiles.remove(name)
    }

    pub fn primary_llm_profile(&self) -> Option<(&String, &LlmConfig)> {
        self.llm_profiles.iter().next()
    }

    pub fn get_embedding_profile(&self, name: &str) -> Option<&EmbeddingConfig> {
        self.embedding_profiles.get(name)
    }

    pub fn upsert_embedding_profile<S: Into<String>>(&mut self, name: S, profile: EmbeddingConfig) {
        self.embedding_profiles.insert(name.into(), profile);
    }

    pub fn remove_embedding_profile(&mut self, name: &str) -> Option<EmbeddingConfig> {
        self.embedding_profiles.remove(name)
    }

    pub fn primary_embedding_profile(&self) -> Option<(&String, &EmbeddingConfig)> {
        self.embedding_profiles.iter().next()
    }

    pub fn from_json_str(input: &str) -> Result<Self, ConfigError> {
        if input.trim().is_empty() {
            return Ok(Self::default());
        }

        let value: Value = serde_json::from_str(input)?;
        Self::from_value(value)
    }

    pub fn from_value(value: Value) -> Result<Self, ConfigError> {
        if value.get("llm_profiles").is_some()
            || value.get("embedding_profiles").is_some()
            || value.get("recent").is_some()
            || value.get("novel").is_some()
        {
            Ok(serde_json::from_value(value)?)
        } else {
            let legacy: LegacyConfig = serde_json::from_value(value)?;
            Ok(Self::from_legacy(legacy))
        }
    }

    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        let data = fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    pub fn to_path(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let serialized = serde_json::to_string_pretty(self)?;
        fs::write(path, serialized)?;
        Ok(())
    }

    fn from_legacy(legacy: LegacyConfig) -> Self {
        let mut config = Self::default();

        if let Some(profile) = legacy.llm_profile() {
            let name = profile
                .interface_format
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "default".to_string());
            config.recent.last_llm_interface = Some(name.clone());
            config.llm_profiles.insert(name, profile.into());
        }

        if let Some((name, profile)) = legacy.embedding_profile() {
            config.recent.last_embedding_interface = Some(name.clone());
            config.embedding_profiles.insert(name, profile);
        }

        config.novel = NovelConfig {
            topic: legacy.topic.clone().unwrap_or_default(),
            genre: legacy.genre.clone().unwrap_or_default(),
            num_chapters: legacy.num_chapters.unwrap_or_default(),
            word_number: legacy.word_number.unwrap_or_default(),
            filepath: legacy.filepath.clone().unwrap_or_default(),
        };

        config
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct LegacyConfig {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    interface_format: Option<String>,
    #[serde(default)]
    model_name: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default)]
    embedding_api_key: Option<String>,
    #[serde(default)]
    embedding_interface_format: Option<String>,
    #[serde(default)]
    embedding_url: Option<String>,
    #[serde(default)]
    embedding_model_name: Option<String>,
    #[serde(default)]
    embedding_retrieval_k: Option<u32>,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    genre: Option<String>,
    #[serde(default)]
    num_chapters: Option<u32>,
    #[serde(default)]
    word_number: Option<u32>,
    #[serde(default)]
    filepath: Option<String>,
}

impl LegacyConfig {
    fn llm_profile(&self) -> Option<LegacyLlmProfile> {
        if self.api_key.is_none()
            && self.base_url.is_none()
            && self.interface_format.is_none()
            && self.model_name.is_none()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.timeout.is_none()
        {
            return None;
        }

        Some(LegacyLlmProfile {
            api_key: self.api_key.clone().unwrap_or_default(),
            base_url: self.base_url.clone().unwrap_or_default(),
            interface_format: self.interface_format.clone(),
            model_name: self.model_name.clone().unwrap_or_default(),
            temperature: self.temperature.unwrap_or_else(default_temperature),
            max_tokens: self.max_tokens.unwrap_or_else(default_max_tokens),
            timeout: self.timeout.unwrap_or_else(default_timeout),
        })
    }

    fn embedding_profile(&self) -> Option<(String, EmbeddingConfig)> {
        let interface_format = self
            .embedding_interface_format
            .clone()
            .or_else(|| self.interface_format.clone());

        if self.embedding_api_key.is_none()
            && self.embedding_url.is_none()
            && interface_format.is_none()
            && self.embedding_model_name.is_none()
            && self.embedding_retrieval_k.is_none()
        {
            return None;
        }

        let name = interface_format
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "default".to_string());

        let profile = EmbeddingConfig {
            api_key: self.embedding_api_key.clone().unwrap_or_default(),
            base_url: self.embedding_url.clone().unwrap_or_default(),
            interface_format: interface_format.unwrap_or_else(|| "default".to_string()),
            model_name: self.embedding_model_name.clone().unwrap_or_default(),
            retrieval_k: self
                .embedding_retrieval_k
                .unwrap_or_else(default_embedding_retrieval_k),
        };

        Some((name, profile))
    }
}

#[derive(Clone, Debug)]
struct LegacyLlmProfile {
    api_key: String,
    base_url: String,
    interface_format: Option<String>,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    timeout: u64,
}

impl From<LegacyLlmProfile> for LlmConfig {
    fn from(profile: LegacyLlmProfile) -> Self {
        Self {
            api_key: profile.api_key,
            base_url: profile.base_url,
            interface_format: profile
                .interface_format
                .unwrap_or_else(|| "default".to_string()),
            model_name: profile.model_name,
            temperature: profile.temperature,
            max_tokens: profile.max_tokens,
            timeout: profile.timeout,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigStore {
    path: PathBuf,
    config: Config,
}

impl ConfigStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ConfigError> {
        let path = path.into();
        let config = if path.exists() {
            Config::from_path(&path)?
        } else {
            Config::default()
        };

        Ok(Self { path, config })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
        Config::from_path(path.as_ref())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    pub fn reload(&mut self) -> Result<(), ConfigError> {
        if self.path.exists() {
            self.config = Config::from_path(&self.path)?;
        } else {
            self.config = Config::default();
        }
        Ok(())
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        self.config.to_path(&self.path)
    }

    pub fn touch_llm_interface<S: Into<String>>(&mut self, name: S) {
        self.config.recent.last_llm_interface = Some(name.into());
    }

    pub fn touch_embedding_interface<S: Into<String>>(&mut self, name: S) {
        self.config.recent.last_embedding_interface = Some(name.into());
    }

    pub fn last_llm_interface(&self) -> Option<&str> {
        self.config
            .recent
            .last_llm_interface
            .as_deref()
            .and_then(|name| self.config.llm_profiles.get(name).map(|_| name))
    }

    pub fn last_embedding_interface(&self) -> Option<&str> {
        self.config
            .recent
            .last_embedding_interface
            .as_deref()
            .and_then(|name| self.config.embedding_profiles.get(name).map(|_| name))
    }

    pub fn ensure_recent_defaults(&mut self) {
        if self
            .config
            .recent
            .last_llm_interface
            .as_ref()
            .map(|name| self.config.llm_profiles.contains_key(name))
            != Some(true)
        {
            let next = self.config.llm_profiles.keys().next().cloned();
            self.config.recent.last_llm_interface = next;
        }

        if self
            .config
            .recent
            .last_embedding_interface
            .as_ref()
            .map(|name| self.config.embedding_profiles.contains_key(name))
            != Some(true)
        {
            let next = self.config.embedding_profiles.keys().next().cloned();
            self.config.recent.last_embedding_interface = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn parses_modern_config() {
        let json = r#"{
            "llm_profiles": {
                "openai": {
                    "api_key": "123",
                    "base_url": "https://api.openai.com/v1",
                    "interface_format": "OpenAI",
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "timeout": 600
                }
            },
            "embedding_profiles": {
                "default": {
                    "api_key": "emb-key",
                    "base_url": "https://api.openai.com/v1",
                    "interface_format": "OpenAI",
                    "model_name": "text-embedding-ada-002",
                    "retrieval_k": 4
                }
            },
            "novel": {
                "topic": "topic",
                "genre": "genre",
                "num_chapters": 10,
                "word_number": 2000,
                "filepath": "path"
            },
            "recent": {
                "last_llm_interface": "openai",
                "last_embedding_interface": "default"
            }
        }"#;

        let config = Config::from_json_str(json).unwrap();
        assert_eq!(config.recent.last_llm_interface.as_deref(), Some("openai"));
        assert_eq!(config.llm_profiles.len(), 1);
    }

    #[test]
    fn parses_legacy_config() {
        let json = r#"{
            "api_key": "abc",
            "base_url": "https://api.openai.com/v1",
            "interface_format": "OpenAI",
            "model_name": "gpt-4o-mini",
            "temperature": 0.8,
            "max_tokens": 4096,
            "embedding_api_key": "emb",
            "embedding_interface_format": "OpenAI",
            "embedding_url": "https://api.openai.com/v1",
            "embedding_model_name": "text-embedding-ada-002",
            "embedding_retrieval_k": 6,
            "topic": "topic",
            "genre": "genre",
            "num_chapters": 12,
            "word_number": 3000,
            "filepath": "path"
        }"#;

        let config = Config::from_json_str(json).unwrap();
        assert_eq!(config.llm_profiles.len(), 1);
        assert_eq!(config.embedding_profiles.len(), 1);
        assert_eq!(config.recent.last_llm_interface.as_deref(), Some("OpenAI"));
        assert_eq!(
            config.recent.last_embedding_interface.as_deref(),
            Some("OpenAI")
        );
    }

    #[test]
    fn store_persists_config() {
        let temp = tempdir().unwrap();
        let config_path = temp.path().join("config.json");

        let mut store = ConfigStore::open(config_path.clone()).unwrap();
        store.config_mut().upsert_llm_profile(
            "openai",
            LlmConfig {
                api_key: "123".into(),
                base_url: "https://api.openai.com/v1".into(),
                interface_format: "OpenAI".into(),
                model_name: "gpt-4o-mini".into(),
                temperature: 0.7,
                max_tokens: 1024,
                timeout: 600,
            },
        );
        store.touch_llm_interface("openai");
        store.save().unwrap();

        let store = ConfigStore::open(config_path).unwrap();
        assert_eq!(store.last_llm_interface(), Some("openai"));
        assert!(store.config().llm_profiles.contains_key("openai"));
    }

    #[test]
    fn ensure_recent_defaults_backfills_missing_profiles() {
        let mut store = ConfigStore::open(PathBuf::from("/nonexistent/config.json")).unwrap();
        store
            .config_mut()
            .upsert_llm_profile("openai", LlmConfig::default());
        store.ensure_recent_defaults();
        assert_eq!(store.last_llm_interface(), Some("openai"));
    }
}
