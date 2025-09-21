use clap::{Args, Parser, Subcommand};
use novel_adapters::{
    create_embedding_adapter, create_llm_adapter, import_knowledge_file, load_vector_store,
    update_vector_store, AdapterError, EmbeddingModel, EmbeddingModelError, LanguageModelError,
    VectorStoreConfig,
};
use novel_core::architecture::ARCHITECTURE_FILE_NAME;
use novel_core::{
    ArchitectureError, ArchitectureRequest, ArchitectureService, BlueprintError, ChapterBlueprint,
    ChapterBlueprintRequest, ChapterBlueprintService, ChapterError, ChapterFinalizer,
    ChapterPromptRequest, ChapterService, ConfigStore, FinalizeChapterRequest, FinalizeError,
    KnowledgeBase, LogLevel, LogRecord, LogSink, PromptError, PromptRegistry, StdoutLogSink,
    BLUEPRINT_FILE_NAME,
};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    let cli = Cli::parse();
    let sink = StdoutLogSink::new();

    match cli.command {
        Command::Config(command) => handle_config(&cli.config, command, &sink),
        Command::Architecture(command) => handle_architecture(&cli.config, command, &sink),
        Command::Blueprint(command) => handle_blueprint(&cli.config, command, &sink),
        Command::Chapter(command) => handle_chapter(&cli.config, command, &sink),
        Command::Knowledge(command) => handle_knowledge(&cli.config, command, &sink),
    }
}

fn handle_config(
    config_path: &Path,
    command: ConfigCommand,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    match command {
        ConfigCommand::TestLlm(args) => run_test_llm(config_path, args, sink),
        ConfigCommand::TestEmbedding(args) => run_test_embedding(config_path, args, sink),
    }
}

fn handle_architecture(
    config_path: &Path,
    command: ArchitectureCommand,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    match command {
        ArchitectureCommand::Generate(args) => run_generate_architecture(config_path, args, sink),
    }
}

fn handle_blueprint(
    config_path: &Path,
    command: BlueprintCommand,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    match command {
        BlueprintCommand::Generate(args) => run_generate_blueprint(config_path, args, sink),
    }
}

fn handle_chapter(
    config_path: &Path,
    command: ChapterCommand,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    match command {
        ChapterCommand::Draft(args) => run_generate_chapter(config_path, args, sink),
        ChapterCommand::Finalize(args) => run_finalize_chapter(config_path, args, sink),
    }
}

fn handle_knowledge(
    config_path: &Path,
    command: KnowledgeCommand,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    match command {
        KnowledgeCommand::Import(args) => run_import_knowledge(config_path, args, sink),
    }
}

fn run_generate_architecture(
    config_path: &Path,
    args: ArchitectureGenerateArgs,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let topic = ensure_novel_field(&store.config().novel.topic, "topic")?;
    let genre = ensure_novel_field(&store.config().novel.genre, "genre")?;
    let chapters = ensure_novel_number(store.config().novel.num_chapters, "num_chapters")?;
    let word_number = ensure_novel_number(store.config().novel.word_number, "word_number")?;
    let output_dir = ensure_output_dir(&store)?;

    let selected_llm = select_llm_interface(&store, args.llm_interface.clone())?;
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始生成世界观架构：{topic}（类型：{genre}，章节数：{chapters}）"),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("输出目录：{}", output_dir.display()),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("使用 LLM 接口：{}", selected_llm),
    ));

    let mut service = ArchitectureService::new(&prompts, sink);
    if let Some(retries) = args.max_retries {
        service = service.with_max_retries(retries);
    }

    let llm_adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let request = ArchitectureRequest {
        topic,
        genre,
        number_of_chapters: chapters,
        word_number,
        user_guidance: args.guidance.unwrap_or_default(),
    };

    let snapshot = service.generate(llm_adapter.as_ref(), &output_dir, &request)?;

    let architecture_path = output_dir.join(ARCHITECTURE_FILE_NAME);
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("小说架构已写入：{}", architecture_path.display()),
    ));
    if snapshot.is_complete() {
        sink.log(LogRecord::new(
            LogLevel::Info,
            "世界观、角色状态与情节架构均已生成完成。".to_string(),
        ));
    }

    store.touch_llm_interface(selected_llm);
    store.save()?;

    Ok(())
}

fn run_generate_blueprint(
    config_path: &Path,
    args: BlueprintGenerateArgs,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let chapters = ensure_novel_number(store.config().novel.num_chapters, "num_chapters")?;
    let output_dir = ensure_output_dir(&store)?;

    let selected_llm = select_llm_interface(&store, args.llm_interface.clone())?;
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始生成章节蓝图，共 {chapters} 章。"),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("使用 LLM 接口：{}", selected_llm),
    ));

    let mut service = ChapterBlueprintService::new(&prompts, sink);
    if let Some(retries) = args.max_retries {
        service = service.with_max_retries(retries);
    }

    let llm_adapter = create_llm_adapter(store.config(), &selected_llm)?;
    let default_max_tokens = store
        .config()
        .get_llm_profile(&selected_llm)
        .map(|profile| profile.max_tokens)
        .unwrap_or(4096);
    let max_tokens = args.max_tokens.unwrap_or(default_max_tokens);
    let request =
        ChapterBlueprintRequest::new(chapters, args.guidance.unwrap_or_default(), max_tokens);

    let blueprint = service.generate(llm_adapter.as_ref(), &output_dir, &request)?;

    let blueprint_path = output_dir.join(BLUEPRINT_FILE_NAME);
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!(
            "章节蓝图已写入：{}（共 {} 章）",
            blueprint_path.display(),
            blueprint.len()
        ),
    ));

    store.touch_llm_interface(selected_llm);
    store.save()?;

    Ok(())
}

fn run_generate_chapter(
    config_path: &Path,
    args: ChapterDraftArgs,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    if args.id == 0 {
        return Err(CliError::InvalidChapterNumber(args.id));
    }

    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let output_dir = ensure_output_dir(&store)?;
    let word_number = ensure_novel_number(store.config().novel.word_number, "word_number")?;

    let selected_llm = select_llm_interface(&store, args.llm_interface.clone())?;
    let selected_embedding =
        select_embedding_interface_optional(&store, args.embedding_interface.clone())?;
    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始生成第{}章草稿。", args.id),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("使用 LLM 接口：{}", selected_llm),
    ));

    let blueprint_path = output_dir.join(BLUEPRINT_FILE_NAME);
    let blueprint_raw = match fs::read_to_string(&blueprint_path) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Err(CliError::MissingBlueprint(blueprint_path))
        }
        Err(source) => {
            return Err(CliError::Io {
                path: blueprint_path,
                source,
            })
        }
    };

    if blueprint_raw.trim().is_empty() {
        return Err(CliError::EmptyBlueprint(blueprint_path));
    }

    let blueprint = ChapterBlueprint::from_text(blueprint_raw);

    let mut request = ChapterPromptRequest::new(&output_dir, &blueprint, args.id, word_number);
    if let Some(guidance) = args.guidance {
        request.user_guidance = guidance;
    }
    if let Some(characters) = args.characters {
        request.characters_involved = characters;
    }
    if let Some(items) = args.items {
        request.key_items = items;
    }
    if let Some(scene) = args.scene {
        request.scene_location = scene;
    }
    if let Some(time) = args.time {
        request.time_constraint = time;
    }
    if let Some(k) = args.retrieval_k {
        request.embedding_retrieval_k = k;
    }
    if let Some(history) = args.history {
        request.history_chapter_count = history.max(1);
    }

    let llm_adapter = create_llm_adapter(store.config(), &selected_llm)?;

    let mut embedding_name_used = None;
    let mut vector_store = None;
    if let Some(name) = selected_embedding.clone() {
        sink.log(LogRecord::new(
            LogLevel::Info,
            format!("使用 Embedding 接口：{}", name),
        ));
        let adapter = create_embedding_adapter(store.config(), &name)?;
        let embedding_arc: Arc<dyn EmbeddingModel> = adapter.into();
        match load_vector_store(sink, Arc::clone(&embedding_arc), &output_dir)? {
            Some(store_instance) => {
                sink.log(LogRecord::new(
                    LogLevel::Info,
                    "已加载向量库上下文，将启用语义检索。".to_string(),
                ));
                vector_store = Some(store_instance);
                embedding_name_used = Some(name);
            }
            None => {
                sink.log(LogRecord::new(
                    LogLevel::Warn,
                    "未检测到向量库元数据，章节将在无检索模式下生成。".to_string(),
                ));
            }
        }
    } else {
        sink.log(LogRecord::new(
            LogLevel::Info,
            "未配置 Embedding 接口，章节将在无知识检索模式下生成。".to_string(),
        ));
    }

    let custom_prompt = if let Some(path) = args.prompt_file.as_ref() {
        sink.log(LogRecord::new(
            LogLevel::Info,
            format!("读取自定义提示词文件：{}", path.display()),
        ));
        Some(fs::read_to_string(path).map_err(|source| CliError::Io {
            path: path.clone(),
            source,
        })?)
    } else {
        None
    };

    let knowledge_base: Option<&dyn KnowledgeBase> = vector_store
        .as_ref()
        .map(|store| store as &dyn KnowledgeBase);

    let chapter_service = ChapterService::new(&prompts, sink);
    let draft = chapter_service.generate_chapter_draft(
        llm_adapter.as_ref(),
        knowledge_base,
        &request,
        custom_prompt.as_deref(),
    )?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("章节草稿已写入：{}", draft.path.display()),
    ));

    store.touch_llm_interface(selected_llm);
    if let Some(name) = embedding_name_used.or(selected_embedding) {
        store.touch_embedding_interface(name);
    }
    store.save()?;

    Ok(())
}

fn run_test_llm(config_path: &Path, args: TestLlmArgs, sink: &dyn LogSink) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let selected = select_llm_interface(&store, args.interface)?;

    let profile = store
        .config()
        .get_llm_profile(&selected)
        .cloned()
        .ok_or_else(|| CliError::UnknownInterface(selected.clone()))?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始测试 LLM 配置：{selected}"),
    ));
    sink.log(LogRecord::new(
        LogLevel::Debug,
        format!(
            "模型: {} | 接口模式: {} | Base URL: {}",
            profile.model_name, profile.interface_format, profile.base_url
        ),
    ));

    let adapter = create_llm_adapter(store.config(), &selected)?;
    sink.log(LogRecord::new(
        LogLevel::Info,
        "发送测试提示词: Please reply 'OK'".to_string(),
    ));

    match adapter.invoke("Please reply 'OK'") {
        Ok(response) => {
            if response.trim().is_empty() {
                sink.log(LogRecord::new(
                    LogLevel::Error,
                    "❌ LLM配置测试失败：未获取到响应".to_string(),
                ));
                return Err(CliError::TestFailed(
                    "LLM配置测试失败：未获取到响应".to_string(),
                ));
            }

            sink.log(LogRecord::new(
                LogLevel::Info,
                "✅ LLM配置测试成功！".to_string(),
            ));
            sink.log(LogRecord::new(
                LogLevel::Debug,
                format!("测试回复: {response}"),
            ));
        }
        Err(err) => {
            sink.log(LogRecord::new(
                LogLevel::Error,
                format!("❌ LLM配置测试出错: {err}"),
            ));
            return Err(CliError::Model(err));
        }
    }

    store.touch_llm_interface(selected);
    store.save()?;

    Ok(())
}

fn run_test_embedding(
    config_path: &Path,
    args: TestEmbeddingArgs,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let selected = select_embedding_interface(&store, args.interface)?;

    let profile = store
        .config()
        .get_embedding_profile(&selected)
        .cloned()
        .ok_or_else(|| CliError::UnknownInterface(selected.clone()))?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始测试 Embedding 配置：{selected}"),
    ));
    sink.log(LogRecord::new(
        LogLevel::Debug,
        format!(
            "模型: {} | 接口模式: {} | Base URL: {}",
            profile.model_name, profile.interface_format, profile.base_url
        ),
    ));

    let adapter = create_embedding_adapter(store.config(), &selected)?;
    sink.log(LogRecord::new(
        LogLevel::Info,
        "发送测试文本: 测试文本".to_string(),
    ));

    match adapter.embed_query("测试文本") {
        Ok(vector) => {
            if vector.is_empty() {
                sink.log(LogRecord::new(
                    LogLevel::Error,
                    "❌ Embedding配置测试失败：未获取到向量".to_string(),
                ));
                return Err(CliError::TestFailed(
                    "Embedding配置测试失败：未获取到向量".to_string(),
                ));
            }

            sink.log(LogRecord::new(
                LogLevel::Info,
                "✅ Embedding配置测试成功！".to_string(),
            ));
            sink.log(LogRecord::new(
                LogLevel::Debug,
                format!("生成的向量维度: {}", vector.len()),
            ));
        }
        Err(err) => {
            sink.log(LogRecord::new(
                LogLevel::Error,
                format!("❌ Embedding配置测试出错: {err}"),
            ));
            return Err(CliError::Embedding(err));
        }
    }

    store.touch_embedding_interface(selected);
    store.save()?;

    Ok(())
}

fn run_finalize_chapter(
    config_path: &Path,
    args: FinalizeArgs,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let output_dir_str = store.config().novel.filepath.trim();
    if output_dir_str.is_empty() {
        return Err(CliError::MissingOutputDir);
    }
    let output_dir = PathBuf::from(output_dir_str);

    let selected_llm = select_llm_interface(&store, args.llm_interface.clone())?;
    let selected_embedding =
        select_embedding_interface_optional(&store, args.embedding_interface.clone())?;

    let prompts = PromptRegistry::from_prompt_config(&store.config().prompts)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始定稿第{}章。", args.id),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("使用 LLM 接口：{}", selected_llm),
    ));

    let llm_adapter = create_llm_adapter(store.config(), &selected_llm)?;

    let embedding_adapter = match selected_embedding.clone() {
        Some(name) => {
            sink.log(LogRecord::new(
                LogLevel::Info,
                format!("使用 Embedding 接口：{}", name),
            ));
            let adapter = create_embedding_adapter(store.config(), &name)?;
            let arc: Arc<dyn EmbeddingModel> = adapter.into();
            Some((name, arc))
        }
        None => {
            sink.log(LogRecord::new(
                LogLevel::Info,
                "未配置 Embedding 接口，向量库更新将跳过。".to_string(),
            ));
            None
        }
    };

    let finalizer = ChapterFinalizer::new(&prompts, sink);
    let embedding_ref: Option<&dyn EmbeddingModel> = embedding_adapter
        .as_ref()
        .map(|(_, adapter)| adapter.as_ref());

    let finalize_request = FinalizeChapterRequest {
        output_dir: output_dir.clone(),
        chapter_number: args.id,
    };
    let result =
        finalizer.finalize_chapter(llm_adapter.as_ref(), embedding_ref, &finalize_request)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("前文摘要已写入：{}", result.summary_path.display()),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("角色状态已写入：{}", result.character_state_path.display()),
    ));
    let mut vector_segments = 0usize;
    if let Some((name, embedding_arc)) = &embedding_adapter {
        match load_vector_store(sink, Arc::clone(embedding_arc), &output_dir)? {
            Some(store) => {
                let chapter_text = fs::read_to_string(&result.chapter_path).map_err(|err| {
                    CliError::Adapter(AdapterError::io(&result.chapter_path, err))
                })?;
                vector_segments = update_vector_store(sink, &store, args.id, &chapter_text)?;
            }
            None => {
                sink.log(LogRecord::new(
                    LogLevel::Warn,
                    "未找到向量库配置，定稿后未执行自动向量写入。".to_string(),
                ));
            }
        }

        sink.log(LogRecord::new(
            LogLevel::Info,
            format!("向量库新增片段数量：{}", vector_segments),
        ));
        store.touch_embedding_interface(name.clone());
    }

    sink.log(LogRecord::new(
        LogLevel::Info,
        "章节定稿流程已完成。".to_string(),
    ));

    store.touch_llm_interface(selected_llm);
    store.save()?;

    Ok(())
}

fn run_import_knowledge(
    config_path: &Path,
    args: KnowledgeImportArgs,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let output_dir_str = store.config().novel.filepath.trim();
    if output_dir_str.is_empty() {
        return Err(CliError::MissingOutputDir);
    }
    let output_dir = PathBuf::from(output_dir_str);

    let selected_embedding = select_embedding_interface(&store, args.embedding_interface.clone())?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始导入知识库文件：{}", args.file.display()),
    ));

    let embedding_adapter = create_embedding_adapter(store.config(), &selected_embedding)?;
    let embedding_arc: Arc<dyn EmbeddingModel> = embedding_adapter.into();

    let config = VectorStoreConfig {
        base_url: args.vector_url.clone(),
        collection_name: args.collection.clone(),
        api_key: args.api_key.clone(),
    };

    let inserted = import_knowledge_file(sink, embedding_arc, &output_dir, config, &args.file)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("知识库导入完成，新增片段数量：{}", inserted),
    ));

    store.touch_embedding_interface(selected_embedding);
    store.save()?;

    Ok(())
}

fn ensure_output_dir(store: &ConfigStore) -> Result<PathBuf, CliError> {
    let output = store.config().novel.filepath.trim();
    if output.is_empty() {
        Err(CliError::MissingOutputDir)
    } else {
        Ok(PathBuf::from(output))
    }
}

fn ensure_novel_field(value: &str, field: &'static str) -> Result<String, CliError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(CliError::MissingNovelField { field })
    } else {
        Ok(trimmed.to_string())
    }
}

fn ensure_novel_number(value: u32, field: &'static str) -> Result<u32, CliError> {
    if value == 0 {
        Err(CliError::MissingNovelNumber { field })
    } else {
        Ok(value)
    }
}

fn select_llm_interface(
    store: &ConfigStore,
    preferred: Option<String>,
) -> Result<String, CliError> {
    if let Some(name) = normalize_preference(preferred) {
        if store.config().llm_profiles.contains_key(&name) {
            return Ok(name);
        }
        return Err(CliError::UnknownInterface(name));
    }

    if let Some(name) = store.last_llm_interface() {
        return Ok(name.to_string());
    }

    if let Some(name) = store.config().llm_profiles.keys().next() {
        return Ok(name.clone());
    }

    Err(CliError::MissingLlmProfile)
}

fn select_embedding_interface(
    store: &ConfigStore,
    preferred: Option<String>,
) -> Result<String, CliError> {
    if let Some(name) = normalize_preference(preferred) {
        if store.config().embedding_profiles.contains_key(&name) {
            return Ok(name);
        }
        return Err(CliError::UnknownInterface(name));
    }

    if let Some(name) = store.last_embedding_interface() {
        return Ok(name.to_string());
    }

    if let Some(name) = store.config().embedding_profiles.keys().next() {
        return Ok(name.clone());
    }

    Err(CliError::MissingEmbeddingProfile)
}

fn select_embedding_interface_optional(
    store: &ConfigStore,
    preferred: Option<String>,
) -> Result<Option<String>, CliError> {
    match select_embedding_interface(store, preferred) {
        Ok(name) => Ok(Some(name)),
        Err(CliError::MissingEmbeddingProfile) => Ok(None),
        Err(other) => Err(other),
    }
}

fn normalize_preference(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

#[derive(Debug, Error)]
enum CliError {
    #[error("配置文件错误: {0}")]
    Config(#[from] novel_core::ConfigError),
    #[error("缺少可用的 LLM 配置，无法执行测试。")]
    MissingLlmProfile,
    #[error("缺少可用的 Embedding 配置，无法执行测试。")]
    MissingEmbeddingProfile,
    #[error("小说输出目录未配置，无法执行该操作。")]
    MissingOutputDir,
    #[error("小说配置 `{field}` 不能为空，请在 config.json 中补全。")]
    MissingNovelField { field: &'static str },
    #[error("小说配置 `{field}` 必须大于 0。")]
    MissingNovelNumber { field: &'static str },
    #[error("章节编号必须从 1 开始，收到 {0}")]
    InvalidChapterNumber(u32),
    #[error("未找到名为 `{0}` 的接口配置")]
    UnknownInterface(String),
    #[error("章节蓝图文件不存在：{0}")]
    MissingBlueprint(PathBuf),
    #[error("章节蓝图文件为空：{0}")]
    EmptyBlueprint(PathBuf),
    #[error("读取文件 `{path}` 失败: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("适配器调用失败: {0}")]
    Adapter(#[from] AdapterError),
    #[error("LLM 调用失败: {0}")]
    Model(#[from] LanguageModelError),
    #[error("Embedding 调用失败: {0}")]
    Embedding(#[from] EmbeddingModelError),
    #[error("提示词加载失败: {0}")]
    Prompt(#[from] PromptError),
    #[error("架构生成失败: {0}")]
    Architecture(#[from] ArchitectureError),
    #[error("章节蓝图生成失败: {0}")]
    Blueprint(#[from] BlueprintError),
    #[error("章节生成失败: {0}")]
    Chapter(#[from] ChapterError),
    #[error("章节定稿失败: {0}")]
    Finalize(#[from] FinalizeError),
    #[error("{0}")]
    TestFailed(String),
}

#[derive(Parser)]
#[command(
    name = "novelctl",
    version,
    about = "AI NovelGenerator 命令行工具 (实验阶段)"
)]
struct Cli {
    /// 指定配置文件路径
    #[arg(long, global = true, default_value = "config.json")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 配置相关操作
    #[command(subcommand)]
    Config(ConfigCommand),
    /// 小说架构相关操作
    #[command(subcommand)]
    Architecture(ArchitectureCommand),
    /// 章节蓝图相关操作
    #[command(subcommand)]
    Blueprint(BlueprintCommand),
    /// 章节相关操作
    #[command(subcommand)]
    Chapter(ChapterCommand),
    /// 知识库相关操作
    #[command(subcommand)]
    Knowledge(KnowledgeCommand),
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// 测试当前 LLM 接口配置
    TestLlm(TestLlmArgs),
    /// 测试当前 Embedding 接口配置
    TestEmbedding(TestEmbeddingArgs),
}

#[derive(Subcommand)]
enum ArchitectureCommand {
    /// 生成或继续补全世界观架构文档
    Generate(ArchitectureGenerateArgs),
}

#[derive(Subcommand)]
enum BlueprintCommand {
    /// 生成或续跑章节蓝图
    Generate(BlueprintGenerateArgs),
}

#[derive(Subcommand)]
enum ChapterCommand {
    /// 生成章节草稿
    Draft(ChapterDraftArgs),
    /// 定稿指定章节，并同步摘要、角色状态及向量库
    Finalize(FinalizeArgs),
}

#[derive(Subcommand)]
enum KnowledgeCommand {
    /// 导入本地知识文件至向量库
    Import(KnowledgeImportArgs),
}

#[derive(Args)]
struct TestLlmArgs {
    /// 指定要测试的接口名称，默认为最近使用的接口
    #[arg(long)]
    interface: Option<String>,
}

#[derive(Args)]
struct TestEmbeddingArgs {
    /// 指定要测试的接口名称，默认为最近使用的接口
    #[arg(long)]
    interface: Option<String>,
}

#[derive(Args)]
struct ArchitectureGenerateArgs {
    /// 指定用于生成架构的 LLM 接口名称
    #[arg(long)]
    llm_interface: Option<String>,
    /// 向模型追加的补充设定
    #[arg(long, value_name = "TEXT")]
    guidance: Option<String>,
    /// 失败重试次数上限，默认使用 3
    #[arg(long, value_name = "N")]
    max_retries: Option<usize>,
}

#[derive(Args)]
struct BlueprintGenerateArgs {
    /// 指定用于生成蓝图的 LLM 接口名称
    #[arg(long)]
    llm_interface: Option<String>,
    /// 向模型追加的目录编排指导
    #[arg(long, value_name = "TEXT")]
    guidance: Option<String>,
    /// 限制单次调用的最大 tokens，用于控制分块大小
    #[arg(long, value_name = "TOKENS")]
    max_tokens: Option<u32>,
    /// 失败重试次数上限，默认使用 3
    #[arg(long, value_name = "N")]
    max_retries: Option<usize>,
}

#[derive(Args)]
struct FinalizeArgs {
    /// 需要定稿的章节编号
    #[arg(long, value_name = "ID")]
    id: u32,
    /// 指定用于定稿的 LLM 接口名称
    #[arg(long)]
    llm_interface: Option<String>,
    /// 指定用于更新向量库的 Embedding 接口名称
    #[arg(long)]
    embedding_interface: Option<String>,
}

#[derive(Args)]
struct ChapterDraftArgs {
    /// 需要生成的章节编号
    #[arg(long, value_name = "ID")]
    id: u32,
    /// 指定用于生成草稿的 LLM 接口名称
    #[arg(long)]
    llm_interface: Option<String>,
    /// 指定用于知识检索的 Embedding 接口名称
    #[arg(long)]
    embedding_interface: Option<String>,
    /// 本章的额外剧情指导
    #[arg(long, value_name = "TEXT")]
    guidance: Option<String>,
    /// 重要角色提示
    #[arg(long, value_name = "TEXT")]
    characters: Option<String>,
    /// 关键道具/线索提示
    #[arg(long, value_name = "TEXT")]
    items: Option<String>,
    /// 场景地点提示
    #[arg(long, value_name = "TEXT")]
    scene: Option<String>,
    /// 时间限制/节奏提示
    #[arg(long, value_name = "TEXT")]
    time: Option<String>,
    /// 知识检索时使用的片段数量
    #[arg(long, value_name = "N")]
    retrieval_k: Option<usize>,
    /// 构建上下文时纳入的历史章节数量
    #[arg(long, value_name = "N")]
    history: Option<usize>,
    /// 外部提示词文件，覆盖自动生成的提示词
    #[arg(long, value_name = "FILE")]
    prompt_file: Option<PathBuf>,
}

#[derive(Args)]
struct KnowledgeImportArgs {
    /// 待导入的知识库文件路径
    #[arg(long, value_name = "FILE")]
    file: PathBuf,
    /// 指定用于生成向量的 Embedding 接口名称
    #[arg(long)]
    embedding_interface: Option<String>,
    /// 向量库服务的 Base URL（例如 http://localhost:6333）
    #[arg(long, default_value = "http://localhost:6333")]
    vector_url: String,
    /// 向量库集合名称，默认为 novel_collection
    #[arg(long, default_value = "novel_collection")]
    collection: String,
    /// 向量库 API Key，可选
    #[arg(long)]
    api_key: Option<String>,
}
