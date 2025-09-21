use clap::{Args, Parser, Subcommand};
use novel_core::{ConfigStore, LogLevel, LogRecord, LogSink, StdoutLogSink};
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
        Command::Config(command) => handle_config(&cli, command, &sink),
    }
}

fn handle_config(cli: &Cli, command: ConfigCommand, sink: &dyn LogSink) -> Result<(), CliError> {
    match command {
        ConfigCommand::TestLlm(args) => run_test_llm(&cli.config, args, sink),
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
        .llm_profiles
        .get(&selected)
        .ok_or_else(|| CliError::UnknownInterface(selected.clone()))?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始测试 LLM 配置（模拟）：{selected}"),
    ));
    sink.log(LogRecord::new(
        LogLevel::Debug,
        format!(
            "模型: {} | 接口模式: {} | Base URL: {}",
            profile.model_name, profile.interface_format, profile.base_url
        ),
    ));
    sink.log(LogRecord::new(
        LogLevel::Info,
        "测试逻辑尚未实现，已完成占位执行。",
    ));

    store.touch_llm_interface(selected);
    store.save()?;

    Ok(())
}

#[derive(Debug, Error)]
enum CliError {
    #[error("配置文件错误: {0}")]
    Config(#[from] novel_core::ConfigError),
    #[error("缺少可用的 LLM 配置，无法执行测试。")]
    MissingLlmProfile,
    #[error("未找到名为 `{0}` 的接口配置")]
    UnknownInterface(String),
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
    Config(ConfigCommand),
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// 测试当前 LLM 接口配置（占位实现）
    TestLlm(TestLlmArgs),
}

#[derive(Args)]
struct TestLlmArgs {
    /// 指定要测试的接口名称，默认为最近使用的接口
    #[arg(long)]
    interface: Option<String>,
}
