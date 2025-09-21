use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use novel_adapters::{
    create_embedding_adapter, create_llm_adapter, import_knowledge_file, load_vector_store,
    update_vector_store, AdapterError, VectorStoreConfig,
};
use novel_core::architecture::{ArchitectureRequest, ArchitectureService};
use novel_core::blueprint::{
    ChapterBlueprint, ChapterBlueprintRequest, ChapterBlueprintService, BLUEPRINT_FILE_NAME,
};
use novel_core::chapter::finalization::{ChapterFinalizer, FinalizeChapterRequest, FinalizeError};
use novel_core::chapter::{ChapterDraft, ChapterPromptRequest, ChapterService, KnowledgeBase};
use novel_core::config::{ConfigError, ConfigStore, NovelConfig};
use novel_core::logging::{LogLevel, LogRecord, LogSink};
use novel_core::prompts::{PromptError, PromptRegistry};
use thiserror::Error;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

pub type EventSender = UnboundedSender<TaskEvent>;

#[derive(Clone, Debug)]
pub enum TaskCommand {
    TestLlm(TestLlmCommand),
    TestEmbedding(TestEmbeddingCommand),
    GenerateArchitecture(GenerateArchitectureCommand),
    GenerateBlueprint(GenerateBlueprintCommand),
    BuildChapterPrompt(BuildChapterPromptCommand),
    GenerateChapterDraft(GenerateChapterDraftCommand),
    FinalizeChapter(FinalizeChapterCommand),
    ImportKnowledge(ImportKnowledgeCommand),
}

impl TaskCommand {
    pub fn kind(&self) -> TaskKind {
        match self {
            TaskCommand::TestLlm(_) => TaskKind::TestLlm,
            TaskCommand::TestEmbedding(_) => TaskKind::TestEmbedding,
            TaskCommand::GenerateArchitecture(_) => TaskKind::GenerateArchitecture,
            TaskCommand::GenerateBlueprint(_) => TaskKind::GenerateBlueprint,
            TaskCommand::BuildChapterPrompt(_) => TaskKind::BuildChapterPrompt,
            TaskCommand::GenerateChapterDraft(_) => TaskKind::GenerateChapterDraft,
            TaskCommand::FinalizeChapter(_) => TaskKind::FinalizeChapter,
            TaskCommand::ImportKnowledge(_) => TaskKind::ImportKnowledge,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskKind {
    TestLlm,
    TestEmbedding,
    GenerateArchitecture,
    GenerateBlueprint,
    BuildChapterPrompt,
    GenerateChapterDraft,
    FinalizeChapter,
    ImportKnowledge,
}

impl TaskKind {
    pub fn label(&self) -> &'static str {
        match self {
            TaskKind::TestLlm => "测试 LLM 配置",
            TaskKind::TestEmbedding => "测试 Embedding 配置",
            TaskKind::GenerateArchitecture => "生成小说架构",
            TaskKind::GenerateBlueprint => "生成章节蓝图",
            TaskKind::BuildChapterPrompt => "构建章节提示词",
            TaskKind::GenerateChapterDraft => "生成章节草稿",
            TaskKind::FinalizeChapter => "章节定稿",
            TaskKind::ImportKnowledge => "导入知识库",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestLlmCommand {
    pub config_path: PathBuf,
    pub interface: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TestEmbeddingCommand {
    pub config_path: PathBuf,
    pub interface: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GenerateArchitectureCommand {
    pub config_path: PathBuf,
    pub output_dir: PathBuf,
    pub topic: String,
    pub genre: String,
    pub number_of_chapters: u32,
    pub word_number: u32,
    pub user_guidance: String,
    pub llm_interface: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GenerateBlueprintCommand {
    pub config_path: PathBuf,
    pub output_dir: PathBuf,
    pub number_of_chapters: u32,
    pub max_tokens: u32,
    pub user_guidance: String,
    pub llm_interface: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BuildChapterPromptCommand {
    pub config_path: PathBuf,
    pub output_dir: PathBuf,
    pub chapter_number: u32,
    pub word_number: u32,
    pub user_guidance: String,
    pub characters_involved: String,
    pub key_items: String,
    pub scene_location: String,
    pub time_constraint: String,
    pub embedding_retrieval_k: usize,
    pub history_chapter_count: usize,
    pub llm_interface: Option<String>,
    pub embedding_interface: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GenerateChapterDraftCommand {
    pub base: BuildChapterPromptCommand,
    pub custom_prompt: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FinalizeChapterCommand {
    pub config_path: PathBuf,
    pub output_dir: PathBuf,
    pub chapter_number: u32,
    pub llm_interface: Option<String>,
    pub embedding_interface: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ImportKnowledgeCommand {
    pub config_path: PathBuf,
    pub output_dir: PathBuf,
    pub file: PathBuf,
    pub embedding_interface: Option<String>,
    pub vector_url: String,
    pub collection: String,
    pub api_key: Option<String>,
}

#[derive(Debug)]
pub struct TaskController {
    sender: UnboundedSender<TaskCommand>,
    receiver: UnboundedReceiver<TaskEvent>,
    _worker: thread::JoinHandle<()>,
}

impl TaskController {
    pub fn new() -> Self {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let worker_tx = event_tx.clone();

        let handle = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build runtime");

            runtime.block_on(async move {
                while let Some(command) = command_rx.recv().await {
                    let sender = worker_tx.clone();
                    tokio::spawn(run_command(command, sender));
                }
            });
        });

        Self {
            sender: command_tx,
            receiver: event_rx,
            _worker: handle,
        }
    }

    pub fn send(&self, command: TaskCommand) -> Result<(), TaskSendError> {
        self.sender
            .send(command)
            .map_err(|_| TaskSendError::ChannelClosed)
    }

    pub fn try_recv(&mut self) -> Option<TaskEvent> {
        self.receiver.try_recv().ok()
    }
}

impl Default for TaskController {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error)]
pub enum TaskSendError {
    #[error("任务通道已关闭")]
    ChannelClosed,
}

#[derive(Debug, Error)]
pub enum TaskError {
    #[error("配置错误: {0}")]
    Config(#[from] ConfigError),
    #[error("适配器错误: {0}")]
    Adapter(#[from] AdapterError),
    #[error("提示词错误: {0}")]
    Prompt(#[from] PromptError),
    #[error("蓝图生成错误: {0}")]
    Blueprint(#[from] novel_core::blueprint::BlueprintError),
    #[error("章节处理错误: {0}")]
    Chapter(#[from] novel_core::chapter::ChapterError),
    #[error("章节定稿错误: {0}")]
    Finalize(#[from] FinalizeError),
    #[error("语言模型错误: {0}")]
    Model(#[from] novel_core::architecture::LanguageModelError),
    #[error("Embedding 错误: {0}")]
    Embedding(#[from] novel_core::embedding::EmbeddingModelError),
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Custom(String),
    #[error("后台任务崩溃: {0}")]
    Join(String),
}

#[derive(Clone, Debug)]
pub enum TaskEvent {
    Log(LogRecord),
    TaskStarted(TaskKind),
    TaskFinished {
        kind: TaskKind,
        result: Result<(), TaskError>,
    },
    ChapterPromptReady {
        number: u32,
        prompt: String,
        summary: String,
        filtered_context: String,
    },
    ChapterDraftReady {
        number: u32,
        path: PathBuf,
        content: String,
        prompt: String,
        summary: String,
        filtered_context: String,
    },
    KnowledgeImportFinished {
        inserted: usize,
    },
}

struct ChannelLogSink {
    sender: EventSender,
}

impl ChannelLogSink {
    fn new(sender: EventSender) -> Self {
        Self { sender }
    }

    fn emit(&self, record: LogRecord) {
        let _ = self.sender.send(TaskEvent::Log(record));
    }
}

impl LogSink for ChannelLogSink {
    fn log(&self, record: LogRecord) {
        self.emit(record);
    }
}

async fn run_command(command: TaskCommand, sender: EventSender) {
    let kind = command.kind();
    let _ = sender.send(TaskEvent::TaskStarted(kind));
    let sender_clone = sender.clone();
    let result = tokio::task::spawn_blocking(move || execute_command(command, sender_clone)).await;
    let outcome = match result {
        Ok(res) => res,
        Err(err) => Err(TaskError::Join(err.to_string())),
    };
    let _ = sender.send(TaskEvent::TaskFinished {
        kind,
        result: outcome,
    });
}

fn execute_command(command: TaskCommand, sender: EventSender) -> Result<(), TaskError> {
    match command {
        TaskCommand::TestLlm(cmd) => run_test_llm(cmd, sender),
        TaskCommand::TestEmbedding(cmd) => run_test_embedding(cmd, sender),
        TaskCommand::GenerateArchitecture(cmd) => run_generate_architecture(cmd, sender),
        TaskCommand::GenerateBlueprint(cmd) => run_generate_blueprint(cmd, sender),
        TaskCommand::BuildChapterPrompt(cmd) => run_build_chapter_prompt(cmd, sender),
        TaskCommand::GenerateChapterDraft(cmd) => run_generate_chapter_draft(cmd, sender),
        TaskCommand::FinalizeChapter(cmd) => run_finalize_chapter(cmd, sender),
        TaskCommand::ImportKnowledge(cmd) => run_import_knowledge(cmd, sender),
    }
}

fn run_test_llm(command: TestLlmCommand, sender: EventSender) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path)?;
    store.ensure_recent_defaults();
    let selected = resolve_llm_interface(&store, command.interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    sink.emit(LogRecord::new(
        LogLevel::Info,
        format!("开始测试 LLM 配置：{selected}"),
    ));
    let profile = store
        .config()
        .get_llm_profile(&selected)
        .cloned()
        .ok_or_else(|| TaskError::Custom("缺少可用的 LLM 配置".to_string()))?;
    sink.emit(LogRecord::new(
        LogLevel::Debug,
        format!(
            "模型: {} | 接口模式: {} | Base URL: {}",
            profile.model_name, profile.interface_format, profile.base_url
        ),
    ));
    let adapter = create_llm_adapter(store.config(), &selected)?;
    sink.emit(LogRecord::new(
        LogLevel::Info,
        "发送测试提示词: Please reply 'OK'".to_string(),
    ));
    let response = adapter.invoke("Please reply 'OK'")?;
    if response.trim().is_empty() {
        sink.emit(LogRecord::new(
            LogLevel::Error,
            "❌ LLM配置测试失败：未获取到响应".to_string(),
        ));
        return Err(TaskError::Custom(
            "LLM配置测试失败：未获取到响应".to_string(),
        ));
    }
    sink.emit(LogRecord::new(
        LogLevel::Info,
        "✅ LLM配置测试成功！".to_string(),
    ));
    sink.emit(LogRecord::new(
        LogLevel::Debug,
        format!("测试回复: {response}"),
    ));
    store.touch_llm_interface(selected);
    store.save()?;
    Ok(())
}

fn run_test_embedding(command: TestEmbeddingCommand, sender: EventSender) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path)?;
    store.ensure_recent_defaults();
    let selected = resolve_embedding_interface(&store, command.interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    sink.emit(LogRecord::new(
        LogLevel::Info,
        format!("开始测试 Embedding 配置：{selected}"),
    ));
    let profile = store
        .config()
        .get_embedding_profile(&selected)
        .cloned()
        .ok_or_else(|| TaskError::Custom("缺少可用的 Embedding 配置".to_string()))?;
    sink.emit(LogRecord::new(
        LogLevel::Debug,
        format!(
            "模型: {} | 接口模式: {} | Base URL: {}",
            profile.model_name, profile.interface_format, profile.base_url
        ),
    ));
    let adapter = create_embedding_adapter(store.config(), &selected)?;
    let vector = adapter.embed_query("测试文本")?;
    if vector.is_empty() {
        sink.emit(LogRecord::new(
            LogLevel::Error,
            "❌ Embedding配置测试失败：未获取到向量".to_string(),
        ));
        return Err(TaskError::Custom(
            "Embedding配置测试失败：未获取到向量".to_string(),
        ));
    }
    sink.emit(LogRecord::new(
        LogLevel::Info,
        format!("✅ Embedding配置测试成功！生成的向量维度: {}", vector.len()),
    ));
    store.touch_embedding_interface(selected);
    store.save()?;
    Ok(())
}

fn run_generate_architecture(
    command: GenerateArchitectureCommand,
    sender: EventSender,
) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path.clone())?;
    store.ensure_recent_defaults();
    let selected_llm = resolve_llm_interface(&store, command.llm_interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    sink.emit(LogRecord::new(
        LogLevel::Info,
        format!("使用 LLM 接口：{selected_llm}"),
    ));
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;
    let service = ArchitectureService::new(&prompts, &sink);
    let adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let request = ArchitectureRequest {
        topic: command.topic.clone(),
        genre: command.genre.clone(),
        number_of_chapters: command.number_of_chapters,
        word_number: command.word_number,
        user_guidance: command.user_guidance.clone(),
    };
    sink.emit(LogRecord::new(
        LogLevel::Info,
        "开始生成小说架构...".to_string(),
    ));
    service.generate(adapter.as_ref(), &command.output_dir, &request)?;
    let mut novel = NovelConfig::default();
    novel.topic = command.topic;
    novel.genre = command.genre;
    novel.num_chapters = command.number_of_chapters;
    novel.word_number = command.word_number;
    novel.filepath = command.output_dir.to_string_lossy().to_string();
    store.config_mut().novel = novel;
    store.touch_llm_interface(selected_llm);
    store.save()?;
    Ok(())
}

fn run_generate_blueprint(
    command: GenerateBlueprintCommand,
    sender: EventSender,
) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path.clone())?;
    store.ensure_recent_defaults();
    let selected_llm = resolve_llm_interface(&store, command.llm_interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    sink.emit(LogRecord::new(
        LogLevel::Info,
        format!("使用 LLM 接口：{selected_llm}"),
    ));
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;
    let service = ChapterBlueprintService::new(&prompts, &sink);
    let adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let request = ChapterBlueprintRequest::new(
        command.number_of_chapters,
        command.user_guidance.clone(),
        command.max_tokens,
    );
    sink.emit(LogRecord::new(
        LogLevel::Info,
        "开始生成章节蓝图...".to_string(),
    ));
    service.generate(adapter.as_ref(), &command.output_dir, &request)?;
    store.config_mut().novel.num_chapters = command.number_of_chapters;
    store.config_mut().novel.filepath = command.output_dir.to_string_lossy().to_string();
    store.touch_llm_interface(selected_llm);
    store.save()?;
    Ok(())
}

fn run_build_chapter_prompt(
    command: BuildChapterPromptCommand,
    sender: EventSender,
) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path.clone())?;
    store.ensure_recent_defaults();
    let selected_llm = resolve_llm_interface(&store, command.llm_interface)?;
    let selected_embedding = resolve_optional_embedding(&store, command.embedding_interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;
    let service = ChapterService::new(&prompts, &sink);
    let adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let embedding = if let Some(name) = &selected_embedding {
        sink.emit(LogRecord::new(
            LogLevel::Info,
            format!("使用 Embedding 接口：{name}"),
        ));
        let adapter = create_embedding_adapter(store.config(), name)?;
        Some((name.clone(), Arc::from(adapter)))
    } else {
        None
    };
    let knowledge = if let Some((_, arc)) = &embedding {
        load_vector_store(&sink, Arc::clone(arc), &command.output_dir)?
    } else {
        None
    };
    let blueprint_path = command.output_dir.join(BLUEPRINT_FILE_NAME);
    let blueprint_text = fs::read_to_string(&blueprint_path)?;
    let blueprint = ChapterBlueprint::from_text(blueprint_text);
    let mut request = ChapterPromptRequest::new(
        &command.output_dir,
        &blueprint,
        command.chapter_number,
        command.word_number,
    );
    request.user_guidance = command.user_guidance.clone();
    request.characters_involved = command.characters_involved.clone();
    request.key_items = command.key_items.clone();
    request.scene_location = command.scene_location.clone();
    request.time_constraint = command.time_constraint.clone();
    request.embedding_retrieval_k = command.embedding_retrieval_k;
    request.history_chapter_count = command.history_chapter_count;
    let knowledge_ref = knowledge.as_ref().map(|store| store as &dyn KnowledgeBase);
    let result = service.build_chapter_prompt(adapter.as_ref(), knowledge_ref, &request)?;
    let _ = sender.send(TaskEvent::ChapterPromptReady {
        number: command.chapter_number,
        prompt: result.prompt_text.clone(),
        summary: result.summary.clone(),
        filtered_context: result.filtered_context.clone(),
    });
    store.touch_llm_interface(selected_llm);
    if let Some((name, _)) = embedding {
        store.touch_embedding_interface(name);
    }
    store.save()?;
    Ok(())
}

fn run_generate_chapter_draft(
    command: GenerateChapterDraftCommand,
    sender: EventSender,
) -> Result<(), TaskError> {
    let base = command.base;
    let mut store = ConfigStore::open(base.config_path.clone())?;
    store.ensure_recent_defaults();
    let selected_llm = resolve_llm_interface(&store, base.llm_interface)?;
    let selected_embedding = resolve_optional_embedding(&store, base.embedding_interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;
    let service = ChapterService::new(&prompts, &sink);
    let adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let embedding = if let Some(name) = &selected_embedding {
        sink.emit(LogRecord::new(
            LogLevel::Info,
            format!("使用 Embedding 接口：{name}"),
        ));
        let adapter = create_embedding_adapter(store.config(), name)?;
        Some((name.clone(), Arc::from(adapter)))
    } else {
        None
    };
    let knowledge = if let Some((_, arc)) = &embedding {
        load_vector_store(&sink, Arc::clone(arc), &base.output_dir)?
    } else {
        None
    };
    let blueprint_path = base.output_dir.join(BLUEPRINT_FILE_NAME);
    let blueprint_text = fs::read_to_string(&blueprint_path)?;
    let blueprint = ChapterBlueprint::from_text(blueprint_text);
    let mut request = ChapterPromptRequest::new(
        &base.output_dir,
        &blueprint,
        base.chapter_number,
        base.word_number,
    );
    request.user_guidance = base.user_guidance.clone();
    request.characters_involved = base.characters_involved.clone();
    request.key_items = base.key_items.clone();
    request.scene_location = base.scene_location.clone();
    request.time_constraint = base.time_constraint.clone();
    request.embedding_retrieval_k = base.embedding_retrieval_k;
    request.history_chapter_count = base.history_chapter_count;
    let knowledge_ref = knowledge.as_ref().map(|store| store as &dyn KnowledgeBase);
    let custom_prompt = command.custom_prompt.as_deref();
    let draft: ChapterDraft =
        service.generate_chapter_draft(adapter.as_ref(), knowledge_ref, &request, custom_prompt)?;
    let _ = sender.send(TaskEvent::ChapterDraftReady {
        number: draft.chapter_number,
        path: draft.path.clone(),
        content: draft.content.clone(),
        prompt: draft.prompt.clone(),
        summary: draft.summary.clone(),
        filtered_context: draft.filtered_context.clone(),
    });
    store.touch_llm_interface(selected_llm);
    if let Some((name, _)) = embedding {
        store.touch_embedding_interface(name);
    }
    store.save()?;
    Ok(())
}

fn run_finalize_chapter(
    command: FinalizeChapterCommand,
    sender: EventSender,
) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path.clone())?;
    store.ensure_recent_defaults();
    let selected_llm = resolve_llm_interface(&store, command.llm_interface)?;
    let selected_embedding = resolve_optional_embedding(&store, command.embedding_interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;
    let finalizer = ChapterFinalizer::new(&prompts, &sink);
    let adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let embedding = if let Some(name) = &selected_embedding {
        sink.emit(LogRecord::new(
            LogLevel::Info,
            format!("使用 Embedding 接口：{name}"),
        ));
        let adapter = create_embedding_adapter(store.config(), name)?;
        Some((name.clone(), Arc::from(adapter)))
    } else {
        None
    };
    let embedding_ref = embedding
        .as_ref()
        .map(|(_, adapter)| adapter.as_ref() as &dyn novel_core::embedding::EmbeddingModel);
    let request = FinalizeChapterRequest {
        output_dir: command.output_dir.clone(),
        chapter_number: command.chapter_number,
    };
    let result = finalizer.finalize_chapter(adapter.as_ref(), embedding_ref, &request)?;
    if let Some((name, arc)) = &embedding {
        if let Some(store_vec) = load_vector_store(&sink, Arc::clone(arc), &command.output_dir)? {
            let chapter_text = fs::read_to_string(&result.chapter_path)?;
            let segments =
                update_vector_store(&sink, &store_vec, command.chapter_number, &chapter_text)?;
            sink.emit(LogRecord::new(
                LogLevel::Info,
                format!("向量库新增片段数量：{}", segments),
            ));
        } else {
            sink.emit(LogRecord::new(
                LogLevel::Warn,
                "未找到向量库配置，定稿后未执行自动向量写入。".to_string(),
            ));
        }
        store.touch_embedding_interface(name.clone());
    }
    store.touch_llm_interface(selected_llm);
    store.save()?;
    Ok(())
}

fn run_import_knowledge(
    command: ImportKnowledgeCommand,
    sender: EventSender,
) -> Result<(), TaskError> {
    let mut store = ConfigStore::open(command.config_path.clone())?;
    store.ensure_recent_defaults();
    let selected_embedding = resolve_embedding_interface(&store, command.embedding_interface)?;
    let sink = ChannelLogSink::new(sender.clone());
    sink.emit(LogRecord::new(
        LogLevel::Info,
        format!("开始导入知识库文件：{}", command.file.display()),
    ));
    let adapter = create_embedding_adapter(store.config(), &selected_embedding)?;
    let arc: Arc<dyn novel_core::embedding::EmbeddingModel> = adapter.into();
    let config = VectorStoreConfig {
        base_url: command.vector_url.clone(),
        collection_name: command.collection.clone(),
        api_key: command.api_key.clone(),
    };
    let inserted = import_knowledge_file(&sink, arc, &command.output_dir, config, &command.file)?;
    let _ = sender.send(TaskEvent::KnowledgeImportFinished { inserted });
    store.touch_embedding_interface(selected_embedding);
    store.save()?;
    Ok(())
}

fn resolve_llm_interface(
    store: &ConfigStore,
    requested: Option<String>,
) -> Result<String, TaskError> {
    if let Some(interface) = requested {
        return Ok(interface);
    }
    if let Some(name) = store.last_llm_interface() {
        return Ok(name.to_string());
    }
    store
        .config()
        .llm_profiles
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| TaskError::Custom("缺少可用的 LLM 配置".to_string()))
}

fn resolve_embedding_interface(
    store: &ConfigStore,
    requested: Option<String>,
) -> Result<String, TaskError> {
    if let Some(interface) = requested {
        return Ok(interface);
    }
    if let Some(name) = store.last_embedding_interface() {
        return Ok(name.to_string());
    }
    store
        .config()
        .embedding_profiles
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| TaskError::Custom("缺少可用的 Embedding 配置".to_string()))
}

fn resolve_optional_embedding(
    store: &ConfigStore,
    requested: Option<String>,
) -> Result<Option<String>, TaskError> {
    if let Some(interface) = requested {
        if interface.trim().is_empty() {
            Ok(None)
        } else {
            Ok(Some(interface))
        }
    } else if let Some(name) = store.last_embedding_interface() {
        Ok(Some(name.to_string()))
    } else {
        Ok(None)
    }
}
