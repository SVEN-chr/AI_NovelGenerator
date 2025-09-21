use clap::{Args, Parser, Subcommand};
use novel_adapters::{
    create_embedding_adapter, create_llm_adapter, AdapterError, EmbeddingModel,
    EmbeddingModelError, LanguageModelError,
};
use novel_core::{
    ChapterFinalizer, ConfigStore, FinalizeChapterRequest, FinalizeError, LogLevel, LogRecord,
    LogSink, PromptError, PromptRegistry, StdoutLogSink,
};
use std::path::{Path, PathBuf};
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
        Command::Chapter(command) => handle_chapter(&cli.config, command, &sink),
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

fn handle_chapter(
    config_path: &Path,
    command: ChapterCommand,
    sink: &dyn LogSink,
) -> Result<(), CliError> {
    match command {
        ChapterCommand::Finalize(args) => run_finalize_chapter(config_path, args, sink),
    }
}

fn run_test_llm(config_path: &Path, args: TestLlmArgs, sink: &dyn LogSink) -> Result<(), CliError> {
    let mut store = ConfigStore::open(config_path.to_path_buf())?;
    store.ensure_recent_defaults();

    let selected = if let Some(interface) = args.interface {
        interface
    } else if let Some(name) = store.last_llm_interface() {
        name.to_string()
    } else if let Some(name) = store.config().llm_profiles.keys().next() {
        name.clone()
    } else {
        return Err(CliError::MissingLlmProfile);
    };

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

    let selected = if let Some(interface) = args.interface {
        interface
    } else if let Some(name) = store.last_embedding_interface() {
        name.to_string()
    } else if let Some(name) = store.config().embedding_profiles.keys().next() {
        name.clone()
    } else {
        return Err(CliError::MissingEmbeddingProfile);
    };

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

    let selected_llm = if let Some(interface) = args.llm_interface.clone() {
        interface
    } else if let Some(name) = store.last_llm_interface() {
        name.to_string()
    } else if let Some(name) = store.config().llm_profiles.keys().next() {
        name.clone()
    } else {
        return Err(CliError::MissingLlmProfile);
    };

    let selected_embedding = if let Some(interface) = args.embedding_interface.clone() {
        Some(interface)
    } else if let Some(name) = store.last_embedding_interface() {
        Some(name.to_string())
    } else if let Some(name) = store.config().embedding_profiles.keys().next() {
        Some(name.clone())
    } else {
        None
    };

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
            Some((name, adapter))
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

    let result = finalizer.finalize_chapter(
        llm_adapter.as_ref(),
        embedding_ref,
        &FinalizeChapterRequest {
            output_dir,
            chapter_number: args.id,
        },
    )?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("前文摘要已写入：{}", result.summary_path.display()),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("角色状态已写入：{}", result.character_state_path.display()),
    ));
    if let Some((_, _)) = &embedding_adapter {
        sink.log(LogRecord::new(
            LogLevel::Info,
            format!("向量库新增片段数量：{}", result.segments_written),
        ));
    }

    sink.log(LogRecord::new(
        LogLevel::Info,
        "章节定稿流程已完成。".to_string(),
    ));

    store.touch_llm_interface(selected_llm);
    if let Some((name, _)) = &embedding_adapter {
        store.touch_embedding_interface(name.clone());
    }
    store.save()?;

    Ok(())
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
    #[error("未找到名为 `{0}` 的接口配置")]
    UnknownInterface(String),
    #[error("适配器调用失败: {0}")]
    Adapter(#[from] AdapterError),
    #[error("LLM 调用失败: {0}")]
    Model(#[from] LanguageModelError),
    #[error("Embedding 调用失败: {0}")]
    Embedding(#[from] EmbeddingModelError),
    #[error("提示词加载失败: {0}")]
    Prompt(#[from] PromptError),
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
    /// 章节相关操作
    #[command(subcommand)]
    Chapter(ChapterCommand),
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// 测试当前 LLM 接口配置
    TestLlm(TestLlmArgs),
    /// 测试当前 Embedding 接口配置
    TestEmbedding(TestEmbeddingArgs),
}

#[derive(Subcommand)]
enum ChapterCommand {
    /// 定稿指定章节，并同步摘要、角色状态及向量库
    Finalize(FinalizeArgs),
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
