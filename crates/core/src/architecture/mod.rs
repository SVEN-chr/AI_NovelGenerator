use crate::logging::{LogLevel, LogRecord, LogSink};
use crate::prompts::{PromptError, PromptRegistry};
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const PARTIAL_FILE_NAME: &str = "partial_architecture.json";
pub const ARCHITECTURE_FILE_NAME: &str = "Novel_architecture.txt";
pub const CHARACTER_STATE_FILE_NAME: &str = "character_state.txt";

#[derive(Debug)]
pub struct LanguageModelError {
    inner: Box<dyn StdError + Send + Sync>,
}

impl LanguageModelError {
    pub fn new<E>(error: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self {
            inner: Box::new(error),
        }
    }

    pub fn into_inner(self) -> Box<dyn StdError + Send + Sync> {
        self.inner
    }

    pub fn as_inner(&self) -> &(dyn StdError + Send + Sync + 'static) {
        self.inner.as_ref()
    }
}

impl fmt::Display for LanguageModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl StdError for LanguageModelError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.inner.as_ref())
    }
}

pub trait LanguageModel: Send + Sync {
    fn invoke(&self, prompt: &str) -> Result<String, LanguageModelError>;
}

#[derive(Debug, Error)]
pub enum ArchitectureError {
    #[error("failed to prepare output directory `{path}`: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read partial architecture from `{path}`: {source}")]
    ReadPartial {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse partial architecture `{path}`: {source}")]
    ParsePartial {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("failed to write partial architecture to `{path}`: {source}")]
    WritePartial {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize partial architecture to `{path}`: {source}")]
    SerializePartial {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("failed to render prompt for stage {stage}: {source}")]
    Prompt {
        stage: ArchitectureStage,
        #[source]
        source: PromptError,
    },
    #[error("language model invocation failed for stage {stage}: {source}")]
    Model {
        stage: ArchitectureStage,
        #[source]
        source: LanguageModelError,
    },
    #[error("stage {stage} returned empty content")]
    EmptyResponse { stage: ArchitectureStage },
    #[error("missing dependency {dependency} when generating stage {stage}")]
    MissingDependency {
        stage: ArchitectureStage,
        dependency: ArchitectureStage,
    },
    #[error("failed to write output file `{path}`: {source}")]
    WriteFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to remove temporary file `{path}`: {source}")]
    RemoveFile {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ArchitectureStage {
    CoreSeed,
    CharacterDynamics,
    CharacterState,
    WorldBuilding,
    PlotArchitecture,
}

impl ArchitectureStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::CoreSeed => "核心种子",
            Self::CharacterDynamics => "角色动力学",
            Self::CharacterState => "角色状态",
            Self::WorldBuilding => "世界观",
            Self::PlotArchitecture => "三幕式情节",
        }
    }

    pub fn dependencies(&self) -> &'static [ArchitectureStage] {
        match self {
            Self::CoreSeed => &[],
            Self::CharacterDynamics => &[ArchitectureStage::CoreSeed],
            Self::CharacterState => &[ArchitectureStage::CharacterDynamics],
            Self::WorldBuilding => &[ArchitectureStage::CoreSeed],
            Self::PlotArchitecture => &[
                ArchitectureStage::CoreSeed,
                ArchitectureStage::CharacterDynamics,
                ArchitectureStage::WorldBuilding,
            ],
        }
    }
}

impl fmt::Display for ArchitectureStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ArchitectureStateData {
    #[serde(
        default,
        rename = "core_seed_result",
        skip_serializing_if = "Option::is_none"
    )]
    core_seed: Option<String>,
    #[serde(
        default,
        rename = "character_dynamics_result",
        skip_serializing_if = "Option::is_none"
    )]
    character_dynamics: Option<String>,
    #[serde(
        default,
        rename = "world_building_result",
        skip_serializing_if = "Option::is_none"
    )]
    world_building: Option<String>,
    #[serde(
        default,
        rename = "plot_arch_result",
        skip_serializing_if = "Option::is_none"
    )]
    plot_architecture: Option<String>,
    #[serde(
        default,
        rename = "character_state_result",
        skip_serializing_if = "Option::is_none"
    )]
    character_state: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ArchitectureState {
    data: ArchitectureStateData,
}

impl ArchitectureState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_from_dir<P: AsRef<Path>>(dir: P, sink: &dyn LogSink) -> Self {
        let path = dir.as_ref().join(PARTIAL_FILE_NAME);
        if !path.exists() {
            return Self::default();
        }

        match fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<ArchitectureStateData>(&contents) {
                Ok(data) => Self { data },
                Err(source) => {
                    sink.log(LogRecord::new(
                        LogLevel::Warn,
                        format!("无法解析 {}，将从空状态开始：{}", path.display(), source),
                    ));
                    Self::default()
                }
            },
            Err(source) => {
                sink.log(LogRecord::new(
                    LogLevel::Warn,
                    format!("无法读取 {}，将从空状态开始：{}", path.display(), source),
                ));
                Self::default()
            }
        }
    }

    pub fn save_to_dir<P: AsRef<Path>>(&self, dir: P) -> Result<(), ArchitectureError> {
        let path = dir.as_ref().join(PARTIAL_FILE_NAME);
        let mut file = File::create(&path).map_err(|source| ArchitectureError::WritePartial {
            path: path.clone(),
            source,
        })?;
        serde_json::to_writer_pretty(&mut file, &self.data).map_err(|source| {
            ArchitectureError::SerializePartial {
                path: path.clone(),
                source,
            }
        })?;
        Ok(())
    }

    pub fn core_seed(&self) -> Option<&str> {
        self.data.core_seed.as_deref()
    }

    pub fn character_dynamics(&self) -> Option<&str> {
        self.data.character_dynamics.as_deref()
    }

    pub fn world_building(&self) -> Option<&str> {
        self.data.world_building.as_deref()
    }

    pub fn plot_architecture(&self) -> Option<&str> {
        self.data.plot_architecture.as_deref()
    }

    pub fn character_state(&self) -> Option<&str> {
        self.data.character_state.as_deref()
    }

    pub fn get(&self, stage: ArchitectureStage) -> Option<&str> {
        match stage {
            ArchitectureStage::CoreSeed => self.core_seed(),
            ArchitectureStage::CharacterDynamics => self.character_dynamics(),
            ArchitectureStage::CharacterState => self.character_state(),
            ArchitectureStage::WorldBuilding => self.world_building(),
            ArchitectureStage::PlotArchitecture => self.plot_architecture(),
        }
    }

    pub fn set(&mut self, stage: ArchitectureStage, value: String) {
        let value = Some(value);
        match stage {
            ArchitectureStage::CoreSeed => self.data.core_seed = value,
            ArchitectureStage::CharacterDynamics => self.data.character_dynamics = value,
            ArchitectureStage::CharacterState => self.data.character_state = value,
            ArchitectureStage::WorldBuilding => self.data.world_building = value,
            ArchitectureStage::PlotArchitecture => self.data.plot_architecture = value,
        }
    }

    pub fn snapshot(&self) -> ArchitectureSnapshot {
        ArchitectureSnapshot {
            core_seed: self.data.core_seed.clone(),
            character_dynamics: self.data.character_dynamics.clone(),
            world_building: self.data.world_building.clone(),
            plot_architecture: self.data.plot_architecture.clone(),
            character_state: self.data.character_state.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArchitectureSnapshot {
    core_seed: Option<String>,
    character_dynamics: Option<String>,
    world_building: Option<String>,
    plot_architecture: Option<String>,
    character_state: Option<String>,
}

impl ArchitectureSnapshot {
    pub fn core_seed(&self) -> Option<&str> {
        self.core_seed.as_deref()
    }

    pub fn character_dynamics(&self) -> Option<&str> {
        self.character_dynamics.as_deref()
    }

    pub fn world_building(&self) -> Option<&str> {
        self.world_building.as_deref()
    }

    pub fn plot_architecture(&self) -> Option<&str> {
        self.plot_architecture.as_deref()
    }

    pub fn character_state(&self) -> Option<&str> {
        self.character_state.as_deref()
    }

    pub fn is_complete(&self) -> bool {
        self.core_seed.is_some()
            && self.character_dynamics.is_some()
            && self.world_building.is_some()
            && self.plot_architecture.is_some()
            && self.character_state.is_some()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArchitectureRequest {
    pub topic: String,
    pub genre: String,
    pub number_of_chapters: u32,
    pub word_number: u32,
    pub user_guidance: String,
}

pub struct ArchitectureService<'a> {
    prompts: &'a PromptRegistry,
    sink: &'a dyn LogSink,
    max_retries: usize,
}

impl<'a> ArchitectureService<'a> {
    pub fn new(prompts: &'a PromptRegistry, sink: &'a dyn LogSink) -> Self {
        Self {
            prompts,
            sink,
            max_retries: 3,
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
        request: &ArchitectureRequest,
    ) -> Result<ArchitectureSnapshot, ArchitectureError> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir).map_err(|source| ArchitectureError::CreateDir {
            path: output_dir.to_path_buf(),
            source,
        })?;

        let mut state = ArchitectureState::load_from_dir(output_dir, self.sink);

        self.generate_core_seed(model, output_dir, request, &mut state)?;
        self.generate_character_dynamics(model, output_dir, request, &mut state)?;
        self.generate_character_state(model, output_dir, &mut state)?;
        self.generate_world_building(model, output_dir, request, &mut state)?;
        self.generate_plot_architecture(model, output_dir, request, &mut state)?;

        self.write_architecture_file(output_dir, request, &state)?;
        self.cleanup_partial(output_dir)?;

        Ok(state.snapshot())
    }

    fn generate_core_seed<M: LanguageModel>(
        &self,
        model: &M,
        output_dir: &Path,
        request: &ArchitectureRequest,
        state: &mut ArchitectureState,
    ) -> Result<(), ArchitectureError> {
        if state.core_seed().is_some() {
            self.log(LogLevel::Info, "Step1 已完成，跳过核心种子生成。");
            return Ok(());
        }

        self.log(LogLevel::Info, "Step1: 生成核心种子...");
        let prompt = self
            .prompts
            .format_with(
                "core_seed",
                [
                    ("topic", request.topic.trim().to_string()),
                    ("genre", request.genre.trim().to_string()),
                    ("number_of_chapters", request.number_of_chapters.to_string()),
                    ("word_number", request.word_number.to_string()),
                    ("user_guidance", request.user_guidance.trim().to_string()),
                ],
            )
            .map_err(|source| ArchitectureError::Prompt {
                stage: ArchitectureStage::CoreSeed,
                source,
            })?;

        let result = self.invoke_with_cleaning(model, ArchitectureStage::CoreSeed, &prompt)?;
        if result.trim().is_empty() {
            state.save_to_dir(output_dir)?;
            self.log(
                LogLevel::Warn,
                "核心种子生成失败，模型返回内容为空。已保存阶段性数据。",
            );
            return Err(ArchitectureError::EmptyResponse {
                stage: ArchitectureStage::CoreSeed,
            });
        }

        state.set(ArchitectureStage::CoreSeed, result);
        state.save_to_dir(output_dir)?;
        Ok(())
    }

    fn generate_character_dynamics<M: LanguageModel>(
        &self,
        model: &M,
        output_dir: &Path,
        request: &ArchitectureRequest,
        state: &mut ArchitectureState,
    ) -> Result<(), ArchitectureError> {
        if state.character_dynamics().is_some() {
            self.log(LogLevel::Info, "Step2 已完成，跳过角色动力学生成。");
            return Ok(());
        }

        self.ensure_dependencies(state, ArchitectureStage::CharacterDynamics)?;

        self.log(LogLevel::Info, "Step2: 生成角色动力学...");
        let prompt = self
            .prompts
            .format_with(
                "character_dynamics",
                [
                    (
                        "core_seed",
                        state.core_seed().unwrap_or_default().trim().to_string(),
                    ),
                    ("user_guidance", request.user_guidance.trim().to_string()),
                ],
            )
            .map_err(|source| ArchitectureError::Prompt {
                stage: ArchitectureStage::CharacterDynamics,
                source,
            })?;

        let result =
            self.invoke_with_cleaning(model, ArchitectureStage::CharacterDynamics, &prompt)?;
        if result.trim().is_empty() {
            state.save_to_dir(output_dir)?;
            self.log(
                LogLevel::Warn,
                "角色动力学生成失败，模型返回内容为空。已保存阶段性数据。",
            );
            return Err(ArchitectureError::EmptyResponse {
                stage: ArchitectureStage::CharacterDynamics,
            });
        }

        state.set(ArchitectureStage::CharacterDynamics, result);
        state.save_to_dir(output_dir)?;
        Ok(())
    }

    fn generate_character_state<M: LanguageModel>(
        &self,
        model: &M,
        output_dir: &Path,
        state: &mut ArchitectureState,
    ) -> Result<(), ArchitectureError> {
        if state.character_state().is_some() {
            self.log(LogLevel::Info, "角色状态已生成，跳过。");
            return Ok(());
        }

        self.ensure_dependencies(state, ArchitectureStage::CharacterState)?;

        self.log(LogLevel::Info, "根据角色动力学生成初始角色状态文档...");

        let prompt = self
            .prompts
            .format_with(
                "create_character_state",
                [(
                    "character_dynamics",
                    state
                        .character_dynamics()
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                )],
            )
            .map_err(|source| ArchitectureError::Prompt {
                stage: ArchitectureStage::CharacterState,
                source,
            })?;

        let result =
            self.invoke_with_cleaning(model, ArchitectureStage::CharacterState, &prompt)?;
        if result.trim().is_empty() {
            state.save_to_dir(output_dir)?;
            self.log(
                LogLevel::Warn,
                "角色状态生成失败，模型返回内容为空。已保存阶段性数据。",
            );
            return Err(ArchitectureError::EmptyResponse {
                stage: ArchitectureStage::CharacterState,
            });
        }

        let character_state_path = output_dir.join(CHARACTER_STATE_FILE_NAME);
        fs::write(&character_state_path, &result).map_err(|source| {
            ArchitectureError::WriteFile {
                path: character_state_path.clone(),
                source,
            }
        })?;
        self.log(
            LogLevel::Info,
            format!("角色状态写入 {} 完成。", character_state_path.display()),
        );

        state.set(ArchitectureStage::CharacterState, result);
        state.save_to_dir(output_dir)?;
        Ok(())
    }

    fn generate_world_building<M: LanguageModel>(
        &self,
        model: &M,
        output_dir: &Path,
        request: &ArchitectureRequest,
        state: &mut ArchitectureState,
    ) -> Result<(), ArchitectureError> {
        if state.world_building().is_some() {
            self.log(LogLevel::Info, "Step3 已完成，跳过世界观生成。");
            return Ok(());
        }

        self.ensure_dependencies(state, ArchitectureStage::WorldBuilding)?;

        self.log(LogLevel::Info, "Step3: 生成世界观...");
        let prompt = self
            .prompts
            .format_with(
                "world_building",
                [
                    (
                        "core_seed",
                        state.core_seed().unwrap_or_default().trim().to_string(),
                    ),
                    ("user_guidance", request.user_guidance.trim().to_string()),
                ],
            )
            .map_err(|source| ArchitectureError::Prompt {
                stage: ArchitectureStage::WorldBuilding,
                source,
            })?;

        let result = self.invoke_with_cleaning(model, ArchitectureStage::WorldBuilding, &prompt)?;
        if result.trim().is_empty() {
            state.save_to_dir(output_dir)?;
            self.log(
                LogLevel::Warn,
                "世界观生成失败，模型返回内容为空。已保存阶段性数据。",
            );
            return Err(ArchitectureError::EmptyResponse {
                stage: ArchitectureStage::WorldBuilding,
            });
        }

        state.set(ArchitectureStage::WorldBuilding, result);
        state.save_to_dir(output_dir)?;
        Ok(())
    }

    fn generate_plot_architecture<M: LanguageModel>(
        &self,
        model: &M,
        output_dir: &Path,
        request: &ArchitectureRequest,
        state: &mut ArchitectureState,
    ) -> Result<(), ArchitectureError> {
        if state.plot_architecture().is_some() {
            self.log(LogLevel::Info, "Step4 已完成，跳过三幕式情节生成。");
            return Ok(());
        }

        self.ensure_dependencies(state, ArchitectureStage::PlotArchitecture)?;

        self.log(LogLevel::Info, "Step4: 生成三幕式情节...");
        let prompt = self
            .prompts
            .format_with(
                "plot_architecture",
                [
                    (
                        "core_seed",
                        state.core_seed().unwrap_or_default().trim().to_string(),
                    ),
                    (
                        "character_dynamics",
                        state
                            .character_dynamics()
                            .unwrap_or_default()
                            .trim()
                            .to_string(),
                    ),
                    (
                        "world_building",
                        state
                            .world_building()
                            .unwrap_or_default()
                            .trim()
                            .to_string(),
                    ),
                    ("user_guidance", request.user_guidance.trim().to_string()),
                ],
            )
            .map_err(|source| ArchitectureError::Prompt {
                stage: ArchitectureStage::PlotArchitecture,
                source,
            })?;

        let result =
            self.invoke_with_cleaning(model, ArchitectureStage::PlotArchitecture, &prompt)?;
        if result.trim().is_empty() {
            state.save_to_dir(output_dir)?;
            self.log(
                LogLevel::Warn,
                "三幕式情节生成失败，模型返回内容为空。已保存阶段性数据。",
            );
            return Err(ArchitectureError::EmptyResponse {
                stage: ArchitectureStage::PlotArchitecture,
            });
        }

        state.set(ArchitectureStage::PlotArchitecture, result);
        state.save_to_dir(output_dir)?;
        Ok(())
    }

    fn ensure_dependencies(
        &self,
        state: &ArchitectureState,
        stage: ArchitectureStage,
    ) -> Result<(), ArchitectureError> {
        for dependency in stage.dependencies() {
            if state.get(*dependency).is_none() {
                return Err(ArchitectureError::MissingDependency {
                    stage,
                    dependency: *dependency,
                });
            }
        }
        Ok(())
    }

    fn invoke_with_cleaning<M: LanguageModel>(
        &self,
        model: &M,
        stage: ArchitectureStage,
        prompt: &str,
    ) -> Result<String, ArchitectureError> {
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
                    let message = err.to_string();
                    self.log(
                        LogLevel::Warn,
                        format!(
                            "LLM 调用失败（{}｜第{}次尝试）：{}",
                            stage.label(),
                            attempt,
                            message
                        ),
                    );
                    if attempt == self.max_retries {
                        return Err(ArchitectureError::Model { stage, source: err });
                    }
                }
            }
        }

        Ok(String::new())
    }

    fn write_architecture_file(
        &self,
        output_dir: &Path,
        request: &ArchitectureRequest,
        state: &ArchitectureState,
    ) -> Result<(), ArchitectureError> {
        let Some(core_seed) = state.core_seed() else {
            return Err(ArchitectureError::MissingDependency {
                stage: ArchitectureStage::PlotArchitecture,
                dependency: ArchitectureStage::CoreSeed,
            });
        };
        let Some(character_dynamics) = state.character_dynamics() else {
            return Err(ArchitectureError::MissingDependency {
                stage: ArchitectureStage::PlotArchitecture,
                dependency: ArchitectureStage::CharacterDynamics,
            });
        };
        let Some(world_building) = state.world_building() else {
            return Err(ArchitectureError::MissingDependency {
                stage: ArchitectureStage::PlotArchitecture,
                dependency: ArchitectureStage::WorldBuilding,
            });
        };
        let Some(plot_architecture) = state.plot_architecture() else {
            return Err(ArchitectureError::MissingDependency {
                stage: ArchitectureStage::PlotArchitecture,
                dependency: ArchitectureStage::PlotArchitecture,
            });
        };

        let final_content = format!(
            "#=== 0) 小说设定 ===\n主题：{topic},类型：{genre},篇幅：约{chapters}章（每章{words}字）\n\n#=== 1) 核心种子 ===\n{core}\n\n#=== 2) 角色动力学 ===\n{characters}\n\n#=== 3) 世界观 ===\n{world}\n\n#=== 4) 三幕式情节架构 ===\n{plot}\n",
            topic = request.topic.trim(),
            genre = request.genre.trim(),
            chapters = request.number_of_chapters,
            words = request.word_number,
            core = core_seed,
            characters = character_dynamics,
            world = world_building,
            plot = plot_architecture,
        );

        let path = output_dir.join(ARCHITECTURE_FILE_NAME);
        fs::write(&path, final_content).map_err(|source| ArchitectureError::WriteFile {
            path: path.clone(),
            source,
        })?;

        self.log(
            LogLevel::Info,
            format!("{} 已生成。", ARCHITECTURE_FILE_NAME),
        );

        Ok(())
    }

    fn cleanup_partial(&self, output_dir: &Path) -> Result<(), ArchitectureError> {
        let path = output_dir.join(PARTIAL_FILE_NAME);
        if !path.exists() {
            return Ok(());
        }

        fs::remove_file(&path).map_err(|source| ArchitectureError::RemoveFile {
            path: path.clone(),
            source,
        })?;
        self.log(
            LogLevel::Info,
            format!("{} 已移除（全部步骤完成）。", PARTIAL_FILE_NAME),
        );
        Ok(())
    }

    fn log(&self, level: LogLevel, message: impl Into<String>) {
        self.sink.log(LogRecord::new(level, message.into()));
    }
}
