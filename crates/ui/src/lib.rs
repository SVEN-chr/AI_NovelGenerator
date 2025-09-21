pub mod app;
pub mod state;
pub mod tasks;
pub mod text_editor;

pub use app::NovelGeneratorApp;
pub use tasks::{TaskCommand, TaskController, TaskEvent, TaskKind};

#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> eframe::Result<()> {
    use eframe::NativeOptions;

    let options = NativeOptions {
        centered: true,
        ..Default::default()
    };
    eframe::run_native(
        "AI Novel Generator",
        options,
        Box::new(|cc| Box::new(NovelGeneratorApp::new(cc))),
    )
}
