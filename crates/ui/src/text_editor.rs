use egui::{Response, TextEdit, Ui};

#[derive(Clone, Debug, Default)]
pub struct TextEditorState {
    text: String,
    has_focus: bool,
}

impl TextEditorState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            has_focus: false,
        }
    }

    pub fn clear(&mut self) {
        self.text.clear();
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn text_mut(&mut self) -> &mut String {
        &mut self.text
    }

    pub fn insert_text(&mut self, snippet: &str) {
        if !self.text.ends_with('\n') && !self.text.is_empty() {
            self.text.push('\n');
        }
        self.text.push_str(snippet);
    }

    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    pub fn ui(&mut self, ui: &mut Ui, id_source: impl std::hash::Hash, rows: usize) -> Response {
        let output = TextEdit::multiline(&mut self.text)
            .id_source(id_source)
            .desired_rows(rows)
            .lock_focus(true)
            .show(ui);

        self.has_focus = output.response.has_focus();
        output.response
    }

    pub fn has_focus(&self) -> bool {
        self.has_focus
    }
}
