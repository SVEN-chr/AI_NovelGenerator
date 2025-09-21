use std::path::{Path, PathBuf};

use eframe::egui::{self, Color32};

use crate::state::{ActiveTab, AppState, ChapterPreviewState, ConfirmationDialog, EditorTarget};
use crate::tasks::{
    TaskCommand, TaskController, TaskEvent, TaskKind, TestEmbeddingCommand, TestLlmCommand,
};
use novel_core::logging::{LogLevel, LogRecord};

pub struct NovelGeneratorApp {
    state: AppState,
    tasks: TaskController,
    status_message: Option<String>,
}

impl NovelGeneratorApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let default_path = PathBuf::from("config.json");
        let mut status_message = None;
        let state = match AppState::new(default_path.clone()) {
            Ok(state) => state,
            Err(err) => {
                let mut message = format!("加载配置失败: {err}。已尝试重置为默认配置。");
                let default = novel_core::config::Config::default();
                if let Err(save_err) = default.to_path(&default_path) {
                    message = format!("加载配置失败: {err}；写入默认配置失败: {save_err}");
                }
                status_message = Some(message);
                AppState::new(default_path).expect("failed to initialize default configuration")
            }
        };

        Self {
            state,
            tasks: TaskController::new(),
            status_message,
        }
    }

    fn handle_event(&mut self, event: TaskEvent) {
        match event {
            TaskEvent::Log(record) => self.state.push_log(record),
            TaskEvent::TaskStarted(kind) => {
                self.state.set_active_task(Some(kind));
            }
            TaskEvent::TaskFinished { kind, result } => {
                self.state.set_active_task(None);
                match result {
                    Ok(()) => {
                        self.state.push_log(LogRecord::new(
                            LogLevel::Info,
                            format!("{} 完成", kind.label()),
                        ));
                        if let Some(err) = self.handle_task_success(kind) {
                            self.status_message = Some(err);
                        } else {
                            self.status_message = Some(format!("{} 完成", kind.label()));
                        }
                    }
                    Err(err) => {
                        self.state.push_log(LogRecord::new(
                            LogLevel::Error,
                            format!("{} 失败: {err}", kind.label()),
                        ));
                        self.status_message = Some(format!("任务失败: {err}"));
                    }
                }
            }
            TaskEvent::ChapterPromptReady {
                number,
                prompt,
                summary,
                filtered_context,
            } => {
                self.state.prompt_editor.set_text(prompt);
                self.state.novel.last_prompt_summary = if summary.trim().is_empty() {
                    None
                } else {
                    Some(summary)
                };
                self.state.novel.last_filtered_context = if filtered_context.trim().is_empty() {
                    None
                } else {
                    Some(filtered_context)
                };
                self.state.push_log(LogRecord::new(
                    LogLevel::Info,
                    format!("章节 {number} 提示词已生成"),
                ));
                self.status_message = Some(format!("章节 {number} 提示词已生成"));
            }
            TaskEvent::ChapterDraftReady {
                number,
                path,
                content,
                prompt,
                summary,
                filtered_context,
            } => {
                self.state.prompt_editor.set_text(prompt);
                self.state.chapter_editor.set_text(content);
                self.state.novel.last_prompt_summary = if summary.trim().is_empty() {
                    None
                } else {
                    Some(summary)
                };
                self.state.novel.last_filtered_context = if filtered_context.trim().is_empty() {
                    None
                } else {
                    Some(filtered_context)
                };
                self.state.push_log(LogRecord::new(
                    LogLevel::Info,
                    format!("章节 {number} 草稿已生成：{}", path.display()),
                ));
                if let Err(err) = self.state.refresh_chapters() {
                    self.status_message = Some(err);
                } else {
                    self.status_message = Some(format!("章节 {number} 草稿已生成"));
                }
            }
            TaskEvent::KnowledgeImportFinished { inserted } => {
                self.state.push_log(LogRecord::new(
                    LogLevel::Info,
                    format!("知识库导入完成，新增片段数量：{inserted}"),
                ));
                self.status_message = Some(format!("知识库导入完成，新增片段数量：{inserted}"));
            }
        }
    }

    fn handle_task_success(&mut self, kind: TaskKind) -> Option<String> {
        let mut errors = Vec::new();
        match kind {
            TaskKind::GenerateArchitecture | TaskKind::GenerateBlueprint => {
                if let Err(err) = self.state.refresh_role_library() {
                    errors.push(err);
                }
                if let Err(err) = self.state.refresh_chapters() {
                    errors.push(err);
                }
            }
            TaskKind::GenerateChapterDraft | TaskKind::FinalizeChapter => {
                if let Err(err) = self.state.refresh_chapters() {
                    errors.push(err);
                }
            }
            _ => {}
        }

        if errors.is_empty() {
            None
        } else {
            Some(errors.join("；"))
        }
    }

    fn save_config(&mut self) {
        match self.state.persist_config() {
            Ok(()) => self.status_message = Some("配置已保存".to_string()),
            Err(err) => self.status_message = Some(format!("保存配置失败: {err}")),
        }
    }

    fn reload_config(&mut self) {
        let trimmed = self.state.config_path_input.trim();
        if trimmed.is_empty() {
            self.status_message = Some("请先输入配置文件路径".to_string());
            return;
        }
        let path = PathBuf::from(trimmed);
        match self.state.reload_from_path(path.clone()) {
            Ok(()) => self.status_message = Some(format!("已重新加载配置：{}", path.display())),
            Err(err) => self.status_message = Some(format!("加载配置失败: {err}")),
        }
    }

    fn dispatch_command(&mut self, command: TaskCommand) {
        if let Err(err) = self.state.persist_config() {
            self.status_message = Some(format!("保存配置失败: {err}"));
            return;
        }
        if let Err(err) = self.tasks.send(command) {
            self.status_message = Some(format!("任务发送失败: {err}"));
        }
    }

    fn show_confirmation_dialog(&mut self, ctx: &egui::Context) {
        if let Some(dialog) = self.state.confirmation.clone() {
            let mut decision = None;
            egui::Window::new(dialog.title.clone())
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(dialog.message.as_str());
                    ui.horizontal(|ui| {
                        if ui.button("确认").clicked() {
                            decision = Some(true);
                        }
                        if ui.button("取消").clicked() {
                            decision = Some(false);
                        }
                    });
                });
            if let Some(confirm) = decision {
                self.state.set_confirmation(None);
                if confirm {
                    self.dispatch_command(dialog.command);
                }
            }
        }
    }

    fn enqueue_confirmation(&mut self, title: &str, message: String, command: TaskCommand) {
        self.state.set_confirmation(Some(ConfirmationDialog {
            title: title.to_string(),
            message,
            command,
        }));
    }
    fn show_config_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("配置文件路径");
            let response = ui.text_edit_singleline(&mut self.state.config_path_input);
            if response.changed() {
                self.status_message = None;
            }
            if ui.button("浏览...").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_file() {
                    self.state.config_path_input = path.display().to_string();
                }
            }
            if ui.button("重新加载").clicked() {
                self.reload_config();
            }
            if ui.button("保存配置").clicked() {
                self.save_config();
            }
        });

        if let Some(status) = &self.status_message {
            ui.colored_label(Color32::LIGHT_BLUE, status);
        }

        ui.separator();
        self.show_llm_section(ui);
        ui.separator();
        self.show_embedding_section(ui);

        ui.separator();
        let busy = self.state.active_task.is_some();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("测试 LLM 配置"))
                .clicked()
            {
                let command = TaskCommand::TestLlm(TestLlmCommand {
                    config_path: self.state.config_store().path().to_path_buf(),
                    interface: self.state.config_panel.selected_llm.clone(),
                });
                self.enqueue_confirmation("确认", "立即测试 LLM 配置？".to_string(), command);
            }
            if ui
                .add_enabled(!busy, egui::Button::new("测试 Embedding 配置"))
                .clicked()
            {
                let command = TaskCommand::TestEmbedding(TestEmbeddingCommand {
                    config_path: self.state.config_store().path().to_path_buf(),
                    interface: self.state.config_panel.selected_embedding.clone(),
                });
                self.enqueue_confirmation("确认", "立即测试 Embedding 配置？".to_string(), command);
            }
        });
    }

    fn show_llm_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("LLM 配置");
        let profiles = self.state.config_panel.llm_profiles.clone();
        ui.horizontal(|ui| {
            let selected = self
                .state
                .config_panel
                .selected_llm
                .clone()
                .unwrap_or_else(|| "未选择".to_string());
            egui::ComboBox::from_label("选择配置")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for name in profiles.iter() {
                        let is_selected =
                            self.state.config_panel.selected_llm.as_ref() == Some(name);
                        if ui.selectable_label(is_selected, name).clicked() {
                            self.state.select_llm_profile(Some(name.clone()));
                        }
                    }
                });
            if ui.button("新增配置").clicked() {
                let name = self.state.config_panel.new_llm_name.clone();
                match self.state.add_llm_profile(&name) {
                    Ok(()) => self.status_message = Some("已新增 LLM 配置".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui.button("删除配置").clicked() {
                if let Some(name) = self.state.config_panel.selected_llm.clone() {
                    match self.state.remove_llm_profile(&name) {
                        Ok(()) => self.status_message = Some(format!("已删除配置 `{name}`")),
                        Err(err) => self.status_message = Some(err),
                    }
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("新增名称");
            ui.text_edit_singleline(&mut self.state.config_panel.new_llm_name);
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("API Key");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.api_key);
        });
        ui.horizontal(|ui| {
            ui.label("Base URL");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.base_url);
        });
        ui.horizontal(|ui| {
            ui.label("接口模式");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.interface_format);
        });
        ui.horizontal(|ui| {
            ui.label("模型名称");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.model_name);
        });
        ui.horizontal(|ui| {
            ui.label("温度");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.temperature);
            ui.label("最大 Token");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.max_tokens);
            ui.label("超时 (s)");
            ui.text_edit_singleline(&mut self.state.config_panel.llm_form.timeout);
        });
    }

    fn show_embedding_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Embedding 配置");
        let profiles = self.state.config_panel.embedding_profiles.clone();
        ui.horizontal(|ui| {
            let selected = self
                .state
                .config_panel
                .selected_embedding
                .clone()
                .unwrap_or_else(|| "未选择".to_string());
            egui::ComboBox::from_label("选择配置")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for name in profiles.iter() {
                        let is_selected =
                            self.state.config_panel.selected_embedding.as_ref() == Some(name);
                        if ui.selectable_label(is_selected, name).clicked() {
                            self.state.select_embedding_profile(Some(name.clone()));
                        }
                    }
                });
            if ui.button("新增配置").clicked() {
                let name = self.state.config_panel.new_embedding_name.clone();
                match self.state.add_embedding_profile(&name) {
                    Ok(()) => self.status_message = Some("已新增 Embedding 配置".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui.button("删除配置").clicked() {
                if let Some(name) = self.state.config_panel.selected_embedding.clone() {
                    match self.state.remove_embedding_profile(&name) {
                        Ok(()) => self.status_message = Some(format!("已删除配置 `{name}`")),
                        Err(err) => self.status_message = Some(err),
                    }
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("新增名称");
            ui.text_edit_singleline(&mut self.state.config_panel.new_embedding_name);
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("API Key");
            ui.text_edit_singleline(&mut self.state.config_panel.embedding_form.api_key);
        });
        ui.horizontal(|ui| {
            ui.label("Base URL");
            ui.text_edit_singleline(&mut self.state.config_panel.embedding_form.base_url);
        });
        ui.horizontal(|ui| {
            ui.label("接口模式");
            ui.text_edit_singleline(&mut self.state.config_panel.embedding_form.interface_format);
        });
        ui.horizontal(|ui| {
            ui.label("模型名称");
            ui.text_edit_singleline(&mut self.state.config_panel.embedding_form.model_name);
        });
        ui.horizontal(|ui| {
            ui.label("检索条目");
            ui.text_edit_singleline(&mut self.state.config_panel.embedding_form.retrieval_k);
        });
    }
    fn show_novel_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("小说基本信息");
        egui::TextEdit::multiline(&mut self.state.novel.topic)
            .desired_rows(3)
            .show(ui);
        ui.horizontal(|ui| {
            ui.label("类型");
            ui.text_edit_singleline(&mut self.state.novel.genre);
            ui.label("章节数");
            ui.text_edit_singleline(&mut self.state.novel.num_chapters);
            ui.label("章节字数");
            ui.text_edit_singleline(&mut self.state.novel.word_number);
        });
        ui.horizontal(|ui| {
            ui.label("输出目录");
            ui.text_edit_singleline(&mut self.state.novel.output_dir);
            if ui.button("浏览...").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.state.novel.output_dir = path.display().to_string();
                }
            }
        });
        if ui.button("保存小说参数").clicked() {
            self.save_config();
        }

        ui.separator();
        ui.heading("章节生成参数");
        ui.horizontal(|ui| {
            ui.label("章节编号");
            ui.text_edit_singleline(&mut self.state.novel.chapter_number);
            ui.label("角色参与");
            ui.text_edit_singleline(&mut self.state.novel.characters_involved);
        });
        ui.horizontal(|ui| {
            ui.label("关键物品");
            ui.text_edit_singleline(&mut self.state.novel.key_items);
            ui.label("场景位置");
            ui.text_edit_singleline(&mut self.state.novel.scene_location);
        });
        ui.horizontal(|ui| {
            ui.label("时间限制");
            ui.text_edit_singleline(&mut self.state.novel.time_constraint);
        });
        ui.horizontal(|ui| {
            ui.label("向量检索 K");
            ui.text_edit_singleline(&mut self.state.novel.embedding_retrieval_k);
            ui.label("历史章节数");
            ui.text_edit_singleline(&mut self.state.novel.history_chapter_count);
        });
        ui.label("内容指导");
        egui::TextEdit::multiline(&mut self.state.novel.user_guidance)
            .desired_rows(4)
            .show(ui);

        ui.separator();
        let busy = self.state.active_task.is_some();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("生成小说架构"))
                .clicked()
            {
                match self.state.make_generate_architecture_command() {
                    Ok(cmd) => self.enqueue_confirmation(
                        "确认",
                        "确定要生成小说架构吗？".to_string(),
                        TaskCommand::GenerateArchitecture(cmd),
                    ),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui
                .add_enabled(!busy, egui::Button::new("生成章节蓝图"))
                .clicked()
            {
                match self.state.make_generate_blueprint_command() {
                    Ok(cmd) => self.enqueue_confirmation(
                        "确认",
                        "确定要生成章节蓝图吗？".to_string(),
                        TaskCommand::GenerateBlueprint(cmd),
                    ),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui
                .add_enabled(!busy, egui::Button::new("构建章节提示词"))
                .clicked()
            {
                match self.state.make_build_prompt_command() {
                    Ok(cmd) => self.enqueue_confirmation(
                        "确认",
                        "构建提示词将调用模型，是否继续？".to_string(),
                        TaskCommand::BuildChapterPrompt(cmd),
                    ),
                    Err(err) => self.status_message = Some(err),
                }
            }
        });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("生成章节草稿"))
                .clicked()
            {
                match self.state.make_generate_draft_command() {
                    Ok(cmd) => self.enqueue_confirmation(
                        "确认",
                        "生成草稿将写入章节文件，是否继续？".to_string(),
                        TaskCommand::GenerateChapterDraft(cmd),
                    ),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui
                .add_enabled(!busy, egui::Button::new("章节定稿"))
                .clicked()
            {
                match self.state.make_finalize_command() {
                    Ok(cmd) => self.enqueue_confirmation(
                        "确认",
                        "定稿将更新摘要与角色状态，是否继续？".to_string(),
                        TaskCommand::FinalizeChapter(cmd),
                    ),
                    Err(err) => self.status_message = Some(err),
                }
            }
        });

        ui.separator();
        ui.heading("知识库导入");
        ui.horizontal(|ui| {
            ui.label("向量库 URL");
            ui.text_edit_singleline(&mut self.state.novel.vector_url);
        });
        ui.horizontal(|ui| {
            ui.label("集合名称");
            ui.text_edit_singleline(&mut self.state.novel.vector_collection);
            ui.label("API Key");
            ui.text_edit_singleline(&mut self.state.novel.vector_api_key);
        });
        ui.horizontal(|ui| {
            ui.label("待导入文件");
            ui.text_edit_singleline(&mut self.state.novel.knowledge_file);
            if ui.button("选择文件").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_file() {
                    self.state.novel.knowledge_file = path.display().to_string();
                }
            }
        });
        if ui
            .add_enabled(!busy, egui::Button::new("导入知识库"))
            .clicked()
        {
            match self.state.make_import_command() {
                Ok(cmd) => self.enqueue_confirmation(
                    "确认",
                    "确定要导入知识库文件吗？".to_string(),
                    TaskCommand::ImportKnowledge(cmd),
                ),
                Err(err) => self.status_message = Some(err),
            }
        }

        ui.separator();
        ui.heading("提示词编辑器");
        let response = self.state.prompt_editor.ui(ui, "prompt_editor", 12);
        self.state
            .update_editor_focus(EditorTarget::Prompt, response.has_focus());
        ui.horizontal(|ui| {
            ui.label(format!("字数：{}", self.state.prompt_editor.char_count()));
            ui.label(format!("词数：{}", self.state.prompt_editor.word_count()));
        });
        if let Some(summary) = &self.state.novel.last_prompt_summary {
            ui.collapsing("最近生成的摘要", |ui| {
                ui.label(summary);
            });
        }
        if let Some(context) = &self.state.novel.last_filtered_context {
            ui.collapsing("过滤后的知识上下文", |ui| {
                ui.label(context);
            });
        }
    }
    fn show_role_library_tab(&mut self, ui: &mut egui::Ui) {
        let Some(output_dir) = self.state.novel.output_dir_path() else {
            ui.label("请先配置小说输出目录。");
            return;
        };

        if self.state.role_library.categories.is_empty() {
            let _ = self.state.refresh_role_library();
        }

        if ui.button("刷新角色库").clicked() {
            if let Err(err) = self.state.refresh_role_library() {
                self.status_message = Some(err);
            }
        }
        if let Some(status) = &self.state.role_library.status {
            ui.colored_label(Color32::LIGHT_BLUE, status);
        }

        ui.columns(2, |columns| {
            self.render_role_sidebar(&output_dir, &mut columns[0]);
            self.render_role_editor(&output_dir, &mut columns[1]);
        });
    }

    fn render_role_sidebar(&mut self, root: &Path, ui: &mut egui::Ui) {
        ui.heading("分类");
        egui::ScrollArea::vertical().show(ui, |ui| {
            let categories = self.state.role_library.categories.clone();
            for category in categories {
                let selected =
                    self.state.role_library.selected_category.as_ref() == Some(&category);
                if ui.selectable_label(selected, category.clone()).clicked() {
                    if let Err(err) = self
                        .state
                        .role_library
                        .select_category(category.clone(), root)
                    {
                        self.status_message = Some(err.to_string());
                    }
                }
            }
        });

        ui.separator();
        ui.heading("角色");
        egui::ScrollArea::vertical().show(ui, |ui| {
            let roles = self.state.role_library.roles.clone();
            for role in roles {
                let selected = self.state.role_library.selected_role.as_ref() == Some(&role.name);
                if ui.selectable_label(selected, role.name.clone()).clicked() {
                    if let Err(err) = self.state.role_library.select_role(role.name.clone(), root) {
                        self.status_message = Some(err.to_string());
                    }
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("新建分类");
            ui.text_edit_singleline(&mut self.state.role_library.new_category_name);
            if ui.button("创建").clicked() {
                let name = self.state.role_library.new_category_name.clone();
                match self.state.role_library.create_category(root, &name) {
                    Ok(()) => self.status_message = Some("已创建分类".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
        });
    }
    fn render_role_editor(&mut self, root: &Path, ui: &mut egui::Ui) {
        ui.heading("角色详情");
        ui.horizontal(|ui| {
            ui.label("角色名称");
            ui.text_edit_singleline(&mut self.state.role_library.role_name_input);
            ui.label("目标分类");
            let categories = self.state.role_library.categories.clone();
            egui::ComboBox::from_id_source("role_move_target")
                .selected_text(self.state.role_library.move_target.clone())
                .show_ui(ui, |ui| {
                    for category in categories {
                        if ui
                            .selectable_label(
                                self.state.role_library.move_target == category,
                                category.clone(),
                            )
                            .clicked()
                        {
                            self.state.role_library.move_target = category;
                        }
                    }
                });
        });

        let response = self.state.role_library.editor.ui(ui, "role_editor", 16);
        self.state
            .update_editor_focus(EditorTarget::Role, response.has_focus());
        ui.horizontal(|ui| {
            ui.label(format!(
                "字数：{}",
                self.state.role_library.editor.char_count()
            ));
        });

        ui.horizontal(|ui| {
            if ui.button("保存").clicked() {
                match self.state.role_library.save(root) {
                    Ok(()) => self.status_message = Some("角色已保存".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui.button("删除").clicked() {
                match self.state.role_library.delete_role(root) {
                    Ok(()) => self.status_message = Some("角色已删除".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui.button("新建角色").clicked() {
                let name = self.state.role_library.role_name_input.clone();
                match self.state.role_library.create_role(root, &name) {
                    Ok(()) => self.status_message = Some("已创建新角色".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
            if ui.button("导入文本").clicked() {
                if let Some(files) = rfd::FileDialog::new().pick_files() {
                    let paths: Vec<PathBuf> = files.into_iter().collect();
                    match self.state.role_library.import_files(root, &paths) {
                        Ok(()) => self.status_message = Some("已导入角色".to_string()),
                        Err(err) => self.status_message = Some(err),
                    }
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("插入到提示词").clicked() {
                let text = self.state.role_library.editor.text().to_string();
                self.state.queue_insert(EditorTarget::Prompt, text);
            }
            if ui.button("插入到章节").clicked() {
                let text = self.state.role_library.editor.text().to_string();
                self.state.queue_insert(EditorTarget::Chapter, text);
            }
        });
    }
    fn show_chapters_tab(&mut self, ui: &mut egui::Ui) {
        if self.state.novel.output_dir_path().is_none() {
            ui.label("请先配置小说输出目录。");
            return;
        }

        ui.horizontal(|ui| {
            if ui.button("刷新").clicked() {
                if let Err(err) = self.state.refresh_chapters() {
                    self.status_message = Some(err);
                }
            }
            if ui.button("上一章").clicked() {
                if let Some(prev) = previous_chapter(&self.state.chapters) {
                    if let Err(err) = self.state.load_chapter(prev) {
                        self.status_message = Some(err);
                    }
                }
            }
            if ui.button("下一章").clicked() {
                if let Some(next) = next_chapter(&self.state.chapters) {
                    if let Err(err) = self.state.load_chapter(next) {
                        self.status_message = Some(err);
                    }
                }
            }
            if ui.button("保存章节").clicked() {
                match self.state.save_current_chapter() {
                    Ok(()) => self.status_message = Some("章节已保存".to_string()),
                    Err(err) => self.status_message = Some(err),
                }
            }
            let chapters = self
                .state
                .chapters
                .chapters
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>();
            let selected_label = self
                .state
                .chapters
                .selected
                .map(|n| n.to_string())
                .unwrap_or_else(|| "未选择".to_string());
            egui::ComboBox::from_label("章节")
                .selected_text(selected_label)
                .show_ui(ui, |ui| {
                    for label in chapters {
                        if let Ok(value) = label.parse::<u32>() {
                            if ui
                                .selectable_label(
                                    self.state.chapters.selected == Some(value),
                                    label,
                                )
                                .clicked()
                            {
                                if let Err(err) = self.state.load_chapter(value) {
                                    self.status_message = Some(err);
                                }
                            }
                        }
                    }
                });
        });

        let response = self.state.chapter_editor.ui(ui, "chapter_editor", 18);
        self.state
            .update_editor_focus(EditorTarget::Chapter, response.has_focus());
        ui.horizontal(|ui| {
            ui.label(format!("字数：{}", self.state.chapter_editor.char_count()));
            ui.label(format!("词数：{}", self.state.chapter_editor.word_count()));
        });
    }
    fn show_logs_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("清空日志").clicked() {
                self.state.clear_logs();
            }
            if let Some(task) = self.state.active_task {
                ui.label(format!("当前任务：{}", task.label()));
            }
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            for record in self.state.logs.iter() {
                let color = match record.level {
                    LogLevel::Error => Color32::RED,
                    LogLevel::Warn => Color32::YELLOW,
                    LogLevel::Info => Color32::LIGHT_GREEN,
                    LogLevel::Debug => Color32::LIGHT_BLUE,
                    LogLevel::Trace => Color32::GRAY,
                };
                ui.colored_label(color, format!("[{}] {}", record.level, record.message));
            }
        });
    }
    fn apply_pending_inserts(&mut self) {
        while let Some((target, text)) = self.state.next_insert() {
            match target {
                EditorTarget::Prompt => self.state.prompt_editor.insert_text(&text),
                EditorTarget::Chapter => self.state.chapter_editor.insert_text(&text),
                EditorTarget::Role => self.state.role_library.editor.insert_text(&text),
            }
        }
    }
}
impl eframe::App for NovelGeneratorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Some(event) = self.tasks.try_recv() {
            self.handle_event(event);
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("AI Novel Generator");
                if let Some(status) = &self.status_message {
                    ui.colored_label(Color32::LIGHT_BLUE, status);
                }
                if let Some(editor) = self.state.focused_editor {
                    ui.label(format!("当前编辑区：{}", editor.label()));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                for tab in ActiveTab::ALL {
                    let selected = self.state.active_tab == tab;
                    if ui.selectable_label(selected, tab.label()).clicked() {
                        self.state.active_tab = tab;
                    }
                }
            });
            ui.separator();
            match self.state.active_tab {
                ActiveTab::Config => self.show_config_tab(ui),
                ActiveTab::Novel => self.show_novel_tab(ui),
                ActiveTab::RoleLibrary => self.show_role_library_tab(ui),
                ActiveTab::Chapters => self.show_chapters_tab(ui),
                ActiveTab::Logs => self.show_logs_tab(ui),
            }
        });

        self.apply_pending_inserts();
        self.show_confirmation_dialog(ctx);
    }
}

fn previous_chapter(state: &ChapterPreviewState) -> Option<u32> {
    let selected = state.selected?;
    let position = state
        .chapters
        .iter()
        .position(|chapter| *chapter == selected)?;
    if position > 0 {
        state.chapters.get(position - 1).copied()
    } else {
        None
    }
}

fn next_chapter(state: &ChapterPreviewState) -> Option<u32> {
    let selected = state.selected?;
    let position = state
        .chapters
        .iter()
        .position(|chapter| *chapter == selected)?;
    state.chapters.get(position + 1).copied()
}
