use crate::tasks::{
    BuildChapterPromptCommand, FinalizeChapterCommand, GenerateArchitectureCommand,
    GenerateBlueprintCommand, GenerateChapterDraftCommand, ImportKnowledgeCommand, TaskCommand,
    TaskKind,
};
use crate::text_editor::TextEditorState;
use novel_core::config::{
    Config, ConfigError, ConfigStore, EmbeddingConfig, LlmConfig, NovelConfig,
};
use novel_core::logging::LogRecord;
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

const ROLE_LIBRARY_DIR: &str = "角色库";
const ROLE_LIBRARY_ALL: &str = "全部";
const CHAPTERS_DIR: &str = "chapters";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActiveTab {
    Config,
    Novel,
    RoleLibrary,
    Chapters,
    Logs,
}

impl ActiveTab {
    pub const ALL: [Self; 5] = [
        ActiveTab::Config,
        ActiveTab::Novel,
        ActiveTab::RoleLibrary,
        ActiveTab::Chapters,
        ActiveTab::Logs,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ActiveTab::Config => "配置面板",
            ActiveTab::Novel => "小说参数",
            ActiveTab::RoleLibrary => "角色库",
            ActiveTab::Chapters => "章节预览",
            ActiveTab::Logs => "日志窗口",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditorTarget {
    Prompt,
    Chapter,
    Role,
}

impl EditorTarget {
    pub fn label(self) -> &'static str {
        match self {
            EditorTarget::Prompt => "提示词编辑器",
            EditorTarget::Chapter => "章节编辑器",
            EditorTarget::Role => "角色编辑器",
        }
    }
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Config(#[from] ConfigError),
}

impl ValidationError {
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }
}

#[derive(Clone)]
pub struct ConfirmationDialog {
    pub title: String,
    pub message: String,
    pub command: TaskCommand,
}

pub struct AppState {
    config_store: ConfigStore,
    pub config_path_input: String,
    pub config_panel: ConfigPanelState,
    pub novel: NovelParametersState,
    pub role_library: RoleLibraryState,
    pub chapters: ChapterPreviewState,
    pub logs: LogPanelState,
    pub prompt_editor: TextEditorState,
    pub chapter_editor: TextEditorState,
    pub active_tab: ActiveTab,
    pub focused_editor: Option<EditorTarget>,
    pending_insert: VecDeque<(EditorTarget, String)>,
    pub confirmation: Option<ConfirmationDialog>,
    pub active_task: Option<TaskKind>,
}

impl AppState {
    pub fn new(config_path: PathBuf) -> Result<Self, ConfigError> {
        let mut store = ConfigStore::open(config_path)?;
        store.ensure_recent_defaults();
        let config_panel = ConfigPanelState::from_store(&store);
        let novel = NovelParametersState::from_config(store.config());
        let role_library = RoleLibraryState::new();
        let chapters = ChapterPreviewState::new();
        let logs = LogPanelState::new();
        let prompt_editor = TextEditorState::new();
        let chapter_editor = TextEditorState::new();
        let config_path_input = store.path().to_string_lossy().to_string();

        Ok(Self {
            config_store: store,
            config_path_input,
            config_panel,
            novel,
            role_library,
            chapters,
            logs,
            prompt_editor,
            chapter_editor,
            active_tab: ActiveTab::Config,
            focused_editor: None,
            pending_insert: VecDeque::new(),
            confirmation: None,
            active_task: None,
        })
    }

    pub fn config_path(&self) -> &Path {
        self.config_store.path()
    }

    pub fn reload_from_path(&mut self, path: PathBuf) -> Result<(), ConfigError> {
        let mut store = ConfigStore::open(path.clone())?;
        store.ensure_recent_defaults();
        self.config_store = store;
        self.config_panel = ConfigPanelState::from_store(&self.config_store);
        self.novel = NovelParametersState::from_config(self.config_store.config());
        self.config_path_input = path.to_string_lossy().to_string();
        Ok(())
    }

    pub fn refresh_from_store(&mut self) {
        self.config_panel = ConfigPanelState::from_store(&self.config_store);
        self.novel = NovelParametersState::from_config(self.config_store.config());
    }

    pub fn sync_form_state(&mut self) -> Result<(), ValidationError> {
        self.config_panel
            .apply_to_store(self.config_store.config_mut())?;
        self.novel.apply_to_store(self.config_store.config_mut())?;
        Ok(())
    }

    pub fn persist_config(&mut self) -> Result<(), ValidationError> {
        self.sync_form_state()?;
        self.config_store.save()?;
        self.refresh_from_store();
        Ok(())
    }

    pub fn config_store(&self) -> &ConfigStore {
        &self.config_store
    }

    pub fn config_store_mut(&mut self) -> &mut ConfigStore {
        &mut self.config_store
    }

    pub fn select_llm_profile(&mut self, name: Option<String>) {
        let store = &self.config_store;
        self.config_panel.select_llm(name, store);
    }

    pub fn select_embedding_profile(&mut self, name: Option<String>) {
        let store = &self.config_store;
        self.config_panel.select_embedding(name, store);
    }

    pub fn queue_insert(&mut self, target: EditorTarget, text: String) {
        if text.trim().is_empty() {
            return;
        }
        self.pending_insert.push_back((target, text));
    }

    pub fn next_insert(&mut self) -> Option<(EditorTarget, String)> {
        self.pending_insert.pop_front()
    }

    pub fn update_editor_focus(&mut self, target: EditorTarget, has_focus: bool) {
        if has_focus {
            self.focused_editor = Some(target);
        } else if self.focused_editor == Some(target) {
            self.focused_editor = None;
        }
    }

    pub fn set_confirmation(&mut self, dialog: Option<ConfirmationDialog>) {
        self.confirmation = dialog;
    }

    pub fn take_confirmation(&mut self) -> Option<ConfirmationDialog> {
        self.confirmation.take()
    }

    pub fn set_active_task(&mut self, task: Option<TaskKind>) {
        self.active_task = task;
    }

    pub fn push_log(&mut self, record: LogRecord) {
        self.logs.push(record);
    }

    pub fn clear_logs(&mut self) {
        self.logs.clear();
    }

    pub fn refresh_role_library(&mut self) -> Result<(), String> {
        let path = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        self.role_library
            .refresh(&path)
            .map_err(|err| format!("角色库读取失败: {err}"))
    }

    pub fn refresh_chapters(&mut self) -> Result<(), String> {
        let path = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        self.chapters
            .refresh(&path)
            .map_err(|err| format!("章节读取失败: {err}"))
    }

    pub fn load_chapter(&mut self, number: u32) -> Result<(), String> {
        let path = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        let text = self
            .chapters
            .load(number, &path)
            .map_err(|err| format!("加载章节失败: {err}"))?;
        self.chapter_editor.set_text(text);
        self.chapters.selected = Some(number);
        Ok(())
    }

    pub fn save_current_chapter(&mut self) -> Result<(), String> {
        let number = self
            .chapters
            .selected
            .ok_or_else(|| "尚未选择章节".to_string())?;
        let path = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        self.chapters
            .save(number, &path, self.chapter_editor.text())
            .map_err(|err| format!("保存章节失败: {err}"))
    }

    pub fn add_llm_profile(&mut self, name: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("请输入新的 LLM 配置名称".to_string());
        }
        if self
            .config_store
            .config()
            .llm_profiles
            .contains_key(trimmed)
        {
            return Err(format!("LLM 配置 `{trimmed}` 已存在"));
        }
        self.config_store
            .config_mut()
            .llm_profiles
            .insert(trimmed.to_string(), LlmConfig::default());
        self.config_panel = ConfigPanelState::from_store(&self.config_store);
        self.config_panel
            .select_llm(Some(trimmed.to_string()), &self.config_store);
        Ok(())
    }

    pub fn remove_llm_profile(&mut self, name: &str) -> Result<(), String> {
        if self
            .config_store
            .config_mut()
            .llm_profiles
            .remove(name)
            .is_none()
        {
            return Err(format!("未找到 LLM 配置 `{name}`"));
        }
        self.config_panel = ConfigPanelState::from_store(&self.config_store);
        Ok(())
    }

    pub fn add_embedding_profile(&mut self, name: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("请输入新的 Embedding 配置名称".to_string());
        }
        if self
            .config_store
            .config()
            .embedding_profiles
            .contains_key(trimmed)
        {
            return Err(format!("Embedding 配置 `{trimmed}` 已存在"));
        }
        self.config_store
            .config_mut()
            .embedding_profiles
            .insert(trimmed.to_string(), EmbeddingConfig::default());
        self.config_panel = ConfigPanelState::from_store(&self.config_store);
        self.config_panel
            .select_embedding(Some(trimmed.to_string()), &self.config_store);
        Ok(())
    }

    pub fn remove_embedding_profile(&mut self, name: &str) -> Result<(), String> {
        if self
            .config_store
            .config_mut()
            .embedding_profiles
            .remove(name)
            .is_none()
        {
            return Err(format!("未找到 Embedding 配置 `{name}`"));
        }
        self.config_panel = ConfigPanelState::from_store(&self.config_store);
        Ok(())
    }

    pub fn make_generate_architecture_command(
        &self,
    ) -> Result<GenerateArchitectureCommand, String> {
        let output_dir = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        let number_of_chapters = self.novel.parse_num_chapters()?;
        let word_number = self.novel.parse_word_number()?;
        let topic = self.novel.topic.clone();
        let genre = self.novel.genre.clone();
        let user_guidance = self.novel.user_guidance.clone();
        Ok(GenerateArchitectureCommand {
            config_path: self.config_store.path().to_path_buf(),
            output_dir,
            topic,
            genre,
            number_of_chapters,
            word_number,
            user_guidance,
            llm_interface: self.config_panel.selected_llm.clone(),
        })
    }

    pub fn make_generate_blueprint_command(&self) -> Result<GenerateBlueprintCommand, String> {
        let output_dir = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        let number_of_chapters = self.novel.parse_num_chapters()?;
        let max_tokens = self.config_panel.llm_form.parse_max_tokens()?;
        let user_guidance = self.novel.user_guidance.clone();
        Ok(GenerateBlueprintCommand {
            config_path: self.config_store.path().to_path_buf(),
            output_dir,
            number_of_chapters,
            max_tokens,
            user_guidance,
            llm_interface: self.config_panel.selected_llm.clone(),
        })
    }

    pub fn make_build_prompt_command(&self) -> Result<BuildChapterPromptCommand, String> {
        let output_dir = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        let chapter_number = self.novel.parse_chapter_number()?;
        let word_number = self.novel.parse_word_number()?;
        let embedding_k = self.novel.parse_embedding_k()?;
        let history = self.novel.parse_history_count()?;
        Ok(BuildChapterPromptCommand {
            config_path: self.config_store.path().to_path_buf(),
            output_dir,
            chapter_number,
            word_number,
            user_guidance: self.novel.user_guidance.clone(),
            characters_involved: self.novel.characters_involved.clone(),
            key_items: self.novel.key_items.clone(),
            scene_location: self.novel.scene_location.clone(),
            time_constraint: self.novel.time_constraint.clone(),
            embedding_retrieval_k: embedding_k,
            history_chapter_count: history,
            llm_interface: self.config_panel.selected_llm.clone(),
            embedding_interface: self.config_panel.selected_embedding.clone(),
        })
    }

    pub fn make_generate_draft_command(&self) -> Result<GenerateChapterDraftCommand, String> {
        let base = self.make_build_prompt_command()?;
        let custom_prompt = if self.prompt_editor.text().trim().is_empty() {
            None
        } else {
            Some(self.prompt_editor.text().to_string())
        };
        Ok(GenerateChapterDraftCommand {
            base,
            custom_prompt,
        })
    }

    pub fn make_finalize_command(&self) -> Result<FinalizeChapterCommand, String> {
        let output_dir = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        let chapter_number = self.novel.parse_chapter_number()?;
        Ok(FinalizeChapterCommand {
            config_path: self.config_store.path().to_path_buf(),
            output_dir,
            chapter_number,
            llm_interface: self.config_panel.selected_llm.clone(),
            embedding_interface: self.config_panel.selected_embedding.clone(),
        })
    }

    pub fn make_import_command(&self) -> Result<ImportKnowledgeCommand, String> {
        let output_dir = self
            .novel
            .output_dir_path()
            .ok_or_else(|| "请先配置小说保存目录".to_string())?;
        let file = self
            .novel
            .knowledge_file_path()
            .ok_or_else(|| "请选择要导入的知识库文件".to_string())?;
        Ok(ImportKnowledgeCommand {
            config_path: self.config_store.path().to_path_buf(),
            output_dir,
            file,
            embedding_interface: self.config_panel.selected_embedding.clone(),
            vector_url: self.novel.vector_url.clone(),
            collection: self.novel.vector_collection.clone(),
            api_key: if self.novel.vector_api_key.trim().is_empty() {
                None
            } else {
                Some(self.novel.vector_api_key.trim().to_string())
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct ConfigPanelState {
    pub selected_llm: Option<String>,
    pub selected_embedding: Option<String>,
    pub llm_profiles: Vec<String>,
    pub embedding_profiles: Vec<String>,
    pub llm_form: LlmProfileForm,
    pub embedding_form: EmbeddingProfileForm,
    pub new_llm_name: String,
    pub new_embedding_name: String,
    pub status: Option<String>,
}

impl ConfigPanelState {
    pub fn from_store(store: &ConfigStore) -> Self {
        let llm_profiles = store
            .config()
            .llm_profiles
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let embedding_profiles = store
            .config()
            .embedding_profiles
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let selected_llm = store.last_llm_interface().map(|s| s.to_string());
        let selected_embedding = store.last_embedding_interface().map(|s| s.to_string());
        let llm_form = selected_llm
            .as_ref()
            .and_then(|name| store.config().llm_profiles.get(name))
            .map(LlmProfileForm::from_config)
            .unwrap_or_default();
        let embedding_form = selected_embedding
            .as_ref()
            .and_then(|name| store.config().embedding_profiles.get(name))
            .map(EmbeddingProfileForm::from_config)
            .unwrap_or_default();
        Self {
            selected_llm,
            selected_embedding,
            llm_profiles,
            embedding_profiles,
            llm_form,
            embedding_form,
            new_llm_name: String::new(),
            new_embedding_name: String::new(),
            status: None,
        }
    }

    pub fn select_llm(&mut self, name: Option<String>, store: &ConfigStore) {
        self.selected_llm = name.clone();
        if let Some(name) = name {
            if let Some(profile) = store.config().llm_profiles.get(&name) {
                self.llm_form = LlmProfileForm::from_config(profile);
            }
        }
    }

    pub fn select_embedding(&mut self, name: Option<String>, store: &ConfigStore) {
        self.selected_embedding = name.clone();
        if let Some(name) = name {
            if let Some(profile) = store.config().embedding_profiles.get(&name) {
                self.embedding_form = EmbeddingProfileForm::from_config(profile);
            }
        }
    }

    pub fn apply_to_store(&self, config: &mut Config) -> Result<(), ValidationError> {
        if let Some(name) = &self.selected_llm {
            let parsed = self.llm_form.to_config()?;
            config.upsert_llm_profile(name.clone(), parsed);
        }
        if let Some(name) = &self.selected_embedding {
            let parsed = self.embedding_form.to_config()?;
            config.upsert_embedding_profile(name.clone(), parsed);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct LlmProfileForm {
    pub api_key: String,
    pub base_url: String,
    pub interface_format: String,
    pub model_name: String,
    pub temperature: String,
    pub max_tokens: String,
    pub timeout: String,
}

impl LlmProfileForm {
    pub fn from_config(config: &LlmConfig) -> Self {
        Self {
            api_key: config.api_key.clone(),
            base_url: config.base_url.clone(),
            interface_format: config.interface_format.clone(),
            model_name: config.model_name.clone(),
            temperature: format!("{:.2}", config.temperature),
            max_tokens: config.max_tokens.to_string(),
            timeout: config.timeout.to_string(),
        }
    }

    pub fn to_config(&self) -> Result<LlmConfig, ValidationError> {
        let temperature: f32 = self
            .temperature
            .parse()
            .map_err(|_| ValidationError::message("温度参数需要为数字"))?;
        let max_tokens: u32 = self
            .max_tokens
            .parse()
            .map_err(|_| ValidationError::message("最大 Token 数需要为整数"))?;
        let timeout: u64 = self
            .timeout
            .parse()
            .map_err(|_| ValidationError::message("请求超时需要为整数"))?;
        Ok(LlmConfig {
            api_key: self.api_key.trim().to_string(),
            base_url: self.base_url.trim().to_string(),
            interface_format: self.interface_format.trim().to_string(),
            model_name: self.model_name.trim().to_string(),
            temperature,
            max_tokens,
            timeout,
        })
    }

    pub fn parse_max_tokens(&self) -> Result<u32, String> {
        self.max_tokens
            .parse()
            .map_err(|_| "最大 Token 数需要为整数".to_string())
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmbeddingProfileForm {
    pub api_key: String,
    pub base_url: String,
    pub interface_format: String,
    pub model_name: String,
    pub retrieval_k: String,
}

impl EmbeddingProfileForm {
    pub fn from_config(config: &EmbeddingConfig) -> Self {
        Self {
            api_key: config.api_key.clone(),
            base_url: config.base_url.clone(),
            interface_format: config.interface_format.clone(),
            model_name: config.model_name.clone(),
            retrieval_k: config.retrieval_k.to_string(),
        }
    }

    pub fn to_config(&self) -> Result<EmbeddingConfig, ValidationError> {
        let retrieval_k: u32 = self
            .retrieval_k
            .parse()
            .map_err(|_| ValidationError::message("检索条目数量需要为整数"))?;
        Ok(EmbeddingConfig {
            api_key: self.api_key.trim().to_string(),
            base_url: self.base_url.trim().to_string(),
            interface_format: self.interface_format.trim().to_string(),
            model_name: self.model_name.trim().to_string(),
            retrieval_k,
        })
    }
}

#[derive(Clone, Debug)]
pub struct NovelParametersState {
    pub topic: String,
    pub genre: String,
    pub num_chapters: String,
    pub word_number: String,
    pub output_dir: String,
    pub chapter_number: String,
    pub user_guidance: String,
    pub characters_involved: String,
    pub key_items: String,
    pub scene_location: String,
    pub time_constraint: String,
    pub embedding_retrieval_k: String,
    pub history_chapter_count: String,
    pub vector_url: String,
    pub vector_collection: String,
    pub vector_api_key: String,
    pub knowledge_file: String,
    pub last_prompt_summary: Option<String>,
    pub last_filtered_context: Option<String>,
}

impl NovelParametersState {
    pub fn from_config(config: &Config) -> Self {
        let novel = &config.novel;
        Self {
            topic: novel.topic.clone(),
            genre: novel.genre.clone(),
            num_chapters: if novel.num_chapters == 0 {
                String::from("10")
            } else {
                novel.num_chapters.to_string()
            },
            word_number: if novel.word_number == 0 {
                String::from("3000")
            } else {
                novel.word_number.to_string()
            },
            output_dir: novel.filepath.clone(),
            chapter_number: String::from("1"),
            user_guidance: String::new(),
            characters_involved: String::new(),
            key_items: String::new(),
            scene_location: String::new(),
            time_constraint: String::new(),
            embedding_retrieval_k: config
                .embedding_profiles
                .values()
                .next()
                .map(|p| p.retrieval_k.to_string())
                .unwrap_or_else(|| "4".to_string()),
            history_chapter_count: String::from("3"),
            vector_url: String::from("http://localhost:6333"),
            vector_collection: String::from("novel_collection"),
            vector_api_key: String::new(),
            knowledge_file: String::new(),
            last_prompt_summary: None,
            last_filtered_context: None,
        }
    }

    pub fn apply_to_store(&self, config: &mut Config) -> Result<(), ValidationError> {
        let num_chapters = self
            .parse_num_chapters()
            .map_err(ValidationError::message)?;
        let word_number = self.parse_word_number().map_err(ValidationError::message)?;
        config.novel = NovelConfig {
            topic: self.topic.trim().to_string(),
            genre: self.genre.trim().to_string(),
            num_chapters,
            word_number,
            filepath: self.output_dir.trim().to_string(),
        };
        Ok(())
    }

    pub fn output_dir_path(&self) -> Option<PathBuf> {
        let trimmed = self.output_dir.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(PathBuf::from(trimmed))
        }
    }

    pub fn knowledge_file_path(&self) -> Option<PathBuf> {
        let trimmed = self.knowledge_file.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(PathBuf::from(trimmed))
        }
    }

    pub fn parse_num_chapters(&self) -> Result<u32, String> {
        self.num_chapters
            .trim()
            .parse()
            .map_err(|_| "章节数量需要为整数".to_string())
    }

    pub fn parse_word_number(&self) -> Result<u32, String> {
        self.word_number
            .trim()
            .parse()
            .map_err(|_| "章节字数需要为整数".to_string())
    }

    pub fn parse_chapter_number(&self) -> Result<u32, String> {
        self.chapter_number
            .trim()
            .parse()
            .map_err(|_| "章节编号需要为整数".to_string())
    }

    pub fn parse_embedding_k(&self) -> Result<usize, String> {
        self.embedding_retrieval_k
            .trim()
            .parse()
            .map_err(|_| "向量检索数量需要为整数".to_string())
    }

    pub fn parse_history_count(&self) -> Result<usize, String> {
        self.history_chapter_count
            .trim()
            .parse()
            .map_err(|_| "历史章节数量需要为整数".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct RoleLibraryState {
    pub categories: Vec<String>,
    pub selected_category: Option<String>,
    pub roles: Vec<RoleEntry>,
    pub selected_role: Option<String>,
    pub role_name_input: String,
    pub new_category_name: String,
    pub move_target: String,
    pub editor: TextEditorState,
    pub status: Option<String>,
}

impl RoleLibraryState {
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            selected_category: None,
            roles: Vec::new(),
            selected_role: None,
            role_name_input: String::new(),
            new_category_name: String::new(),
            move_target: ROLE_LIBRARY_ALL.to_string(),
            editor: TextEditorState::new(),
            status: None,
        }
    }

    pub fn refresh(&mut self, root: &Path) -> io::Result<()> {
        ensure_role_library(root)?;
        self.categories = collect_categories(root);
        if self.selected_category.is_none() {
            self.selected_category = Some(ROLE_LIBRARY_ALL.to_string());
        }
        if !self
            .categories
            .iter()
            .any(|c| Some(c) == self.selected_category.as_ref())
        {
            self.selected_category = Some(ROLE_LIBRARY_ALL.to_string());
        }
        self.update_roles(root)?;
        Ok(())
    }

    fn update_roles(&mut self, root: &Path) -> io::Result<()> {
        let category = self
            .selected_category
            .clone()
            .unwrap_or_else(|| ROLE_LIBRARY_ALL.to_string());
        self.roles = collect_roles(root, &category)?;
        if let Some(selected) = &self.selected_role {
            if !self.roles.iter().any(|role| &role.name == selected) {
                self.selected_role = None;
            }
        }
        if self.selected_role.is_none() {
            self.selected_role = self.roles.first().map(|role| role.name.clone());
        }
        if let Some(selected) = self.selected_role.clone() {
            self.role_name_input = selected.clone();
            if let Some(entry) = self.current_entry() {
                match fs::read_to_string(&entry.path) {
                    Ok(text) => self.editor.set_text(text),
                    Err(err) => {
                        self.editor.clear();
                        self.status = Some(format!("读取角色文件失败: {err}"));
                    }
                }
            }
        } else {
            self.editor.clear();
            self.role_name_input.clear();
        }
        self.move_target = self
            .selected_category
            .clone()
            .unwrap_or_else(|| ROLE_LIBRARY_ALL.to_string());
        Ok(())
    }

    pub fn select_category(&mut self, category: String, root: &Path) -> io::Result<()> {
        self.selected_category = Some(category);
        self.update_roles(root)
    }

    pub fn select_role(&mut self, role: String, root: &Path) -> io::Result<()> {
        self.selected_role = Some(role.clone());
        self.role_name_input = role;
        self.update_roles(root)
    }

    pub fn save(&mut self, root: &Path) -> Result<(), String> {
        if self.role_name_input.trim().is_empty() {
            return Err("请填写角色名称".to_string());
        }
        let entry = self
            .current_entry()
            .cloned()
            .ok_or_else(|| "请先选择角色".to_string())?;
        let new_name = self.role_name_input.trim();
        let target_category = if self.move_target.trim().is_empty() {
            entry.category.clone()
        } else {
            self.move_target.trim().to_string()
        };
        ensure_category(root, &target_category)
            .map_err(|err| format!("创建分类目录失败: {err}"))?;
        let new_path = role_path(root, &target_category, new_name);
        if entry.path != new_path {
            if new_path.exists() {
                return Err("目标分类中已存在同名角色".to_string());
            }
            fs::rename(&entry.path, &new_path).map_err(|err| format!("移动角色文件失败: {err}"))?;
        }
        fs::write(&new_path, self.editor.text())
            .map_err(|err| format!("写入角色文件失败: {err}"))?;
        self.status = Some("角色已保存".to_string());
        self.selected_category = Some(target_category.clone());
        self.selected_role = Some(new_name.to_string());
        self.update_roles(root)
            .map_err(|err| format!("刷新角色列表失败: {err}"))?;
        Ok(())
    }

    pub fn create_category(&mut self, root: &Path, name: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("请输入分类名称".to_string());
        }
        ensure_category(root, trimmed).map_err(|err| err.to_string())?;
        self.refresh(root)
            .map_err(|err| format!("刷新角色列表失败: {err}"))?;
        self.status = Some(format!("已创建分类 `{trimmed}`"));
        Ok(())
    }

    pub fn create_role(&mut self, root: &Path, name: &str) -> Result<(), String> {
        let category = self
            .selected_category
            .clone()
            .unwrap_or_else(|| ROLE_LIBRARY_ALL.to_string());
        let target = if category == ROLE_LIBRARY_ALL {
            ROLE_LIBRARY_ALL.to_string()
        } else {
            category
        };
        ensure_category(root, &target).map_err(|err| format!("创建分类目录失败: {err}"))?;
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("请输入角色名称".to_string());
        }
        let path = role_path(root, &target, trimmed);
        if path.exists() {
            return Err("同名角色已存在".to_string());
        }
        fs::write(&path, "").map_err(|err| format!("创建角色文件失败: {err}"))?;
        self.status = Some("已创建空白角色".to_string());
        self.selected_category = Some(target);
        self.selected_role = Some(trimmed.to_string());
        self.update_roles(root)
            .map_err(|err| format!("刷新角色列表失败: {err}"))?;
        Ok(())
    }

    pub fn delete_role(&mut self, root: &Path) -> Result<(), String> {
        let entry = self
            .current_entry()
            .cloned()
            .ok_or_else(|| "请先选择角色".to_string())?;
        fs::remove_file(&entry.path).map_err(|err| format!("删除角色文件失败: {err}"))?;
        self.status = Some("角色已删除".to_string());
        self.selected_role = None;
        self.update_roles(root)
            .map_err(|err| format!("刷新角色列表失败: {err}"))?;
        Ok(())
    }

    pub fn import_files(&mut self, root: &Path, files: &[PathBuf]) -> Result<(), String> {
        if files.is_empty() {
            return Ok(());
        }
        let category = self
            .selected_category
            .clone()
            .unwrap_or_else(|| ROLE_LIBRARY_ALL.to_string());
        let target = if category == ROLE_LIBRARY_ALL {
            ROLE_LIBRARY_ALL.to_string()
        } else {
            category
        };
        ensure_category(root, &target).map_err(|err| format!("创建分类目录失败: {err}"))?;
        let mut imported = 0usize;
        let mut errors = Vec::new();
        for file in files {
            if let Some(name) = file.file_stem().and_then(|s| s.to_str()) {
                let path = role_path(root, &target, name);
                match fs::copy(file, &path) {
                    Ok(_) => imported += 1,
                    Err(err) => errors.push(format!("导入 `{name}` 失败: {err}")),
                }
            }
        }
        self.update_roles(root)
            .map_err(|err| format!("刷新角色列表失败: {err}"))?;
        self.status = Some(if !errors.is_empty() {
            if imported > 0 {
                format!(
                    "成功导入 {imported} 个角色，但部分失败：{}",
                    errors.join("；")
                )
            } else {
                errors.join("；")
            }
        } else if imported > 0 {
            format!("成功导入 {imported} 个角色")
        } else {
            "未导入任何角色".to_string()
        });
        Ok(())
    }

    fn current_entry(&self) -> Option<&RoleEntry> {
        let selected = self.selected_role.as_ref()?;
        self.roles.iter().find(|entry| &entry.name == selected)
    }
}

#[derive(Clone, Debug)]
pub struct RoleEntry {
    pub name: String,
    pub category: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ChapterPreviewState {
    pub chapters: Vec<u32>,
    pub selected: Option<u32>,
    pub status: Option<String>,
}

impl ChapterPreviewState {
    pub fn new() -> Self {
        Self {
            chapters: Vec::new(),
            selected: None,
            status: None,
        }
    }

    pub fn refresh(&mut self, root: &Path) -> io::Result<()> {
        let chapters_dir = root.join(CHAPTERS_DIR);
        if !chapters_dir.exists() {
            self.chapters.clear();
            self.selected = None;
            return Ok(());
        }
        let mut numbers = Vec::new();
        for entry in fs::read_dir(&chapters_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                if let Some(num) = parse_chapter_filename(file_name) {
                    numbers.push(num);
                }
            }
        }
        numbers.sort_unstable();
        self.chapters = numbers;
        if let Some(selected) = self.selected {
            if !self.chapters.contains(&selected) {
                self.selected = self.chapters.first().copied();
            }
        } else {
            self.selected = self.chapters.first().copied();
        }
        Ok(())
    }

    pub fn load(&mut self, chapter: u32, root: &Path) -> io::Result<String> {
        let path = chapter_path(root, chapter);
        fs::read_to_string(&path)
    }

    pub fn save(&self, chapter: u32, root: &Path, content: &str) -> io::Result<()> {
        let path = chapter_path(root, chapter);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(&path, content)
    }
}

#[derive(Clone, Debug)]
pub struct LogPanelState {
    records: VecDeque<LogRecord>,
    capacity: usize,
}

impl LogPanelState {
    pub fn new() -> Self {
        Self {
            records: VecDeque::new(),
            capacity: 500,
        }
    }

    pub fn push(&mut self, record: LogRecord) {
        if self.records.len() >= self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &LogRecord> {
        self.records.iter()
    }
}

fn ensure_role_library(root: &Path) -> io::Result<()> {
    fs::create_dir_all(root.join(ROLE_LIBRARY_DIR).join(ROLE_LIBRARY_ALL))
}

fn ensure_category(root: &Path, category: &str) -> io::Result<()> {
    fs::create_dir_all(root.join(ROLE_LIBRARY_DIR).join(category))
}

fn collect_categories(root: &Path) -> Vec<String> {
    let mut categories = vec![ROLE_LIBRARY_ALL.to_string()];
    let role_dir = root.join(ROLE_LIBRARY_DIR);
    if let Ok(entries) = fs::read_dir(&role_dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name != ROLE_LIBRARY_ALL {
                        categories.push(name);
                    }
                }
            }
        }
    }
    categories.sort();
    categories
}

fn collect_roles(root: &Path, category: &str) -> io::Result<Vec<RoleEntry>> {
    let mut roles = Vec::new();
    if category == ROLE_LIBRARY_ALL {
        for entry in collect_categories(root) {
            if entry == ROLE_LIBRARY_ALL {
                continue;
            }
            roles.extend(collect_roles(root, &entry)?);
        }
    } else {
        let dir = root.join(ROLE_LIBRARY_DIR).join(category);
        if dir.exists() {
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                        roles.push(RoleEntry {
                            name: name.to_string(),
                            category: category.to_string(),
                            path: entry.path(),
                        });
                    }
                }
            }
        }
    }
    roles.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(roles)
}

fn role_path(root: &Path, category: &str, name: &str) -> PathBuf {
    root.join(ROLE_LIBRARY_DIR)
        .join(category)
        .join(format!("{name}.txt"))
}

fn parse_chapter_filename(name: &str) -> Option<u32> {
    if let Some(stripped) = name.strip_prefix("chapter_") {
        let stripped = stripped.strip_suffix(".txt").unwrap_or(stripped);
        stripped.parse().ok()
    } else {
        None
    }
}

fn chapter_path(root: &Path, chapter: u32) -> PathBuf {
    root.join(CHAPTERS_DIR)
        .join(format!("chapter_{chapter}.txt"))
}
