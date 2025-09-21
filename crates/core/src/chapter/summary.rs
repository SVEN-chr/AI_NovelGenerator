use std::fs;
use std::io;
use std::path::Path;

use crate::architecture::LanguageModel;
use crate::blueprint::ChapterBlueprintEntry;
use crate::prompts::{PromptArguments, PromptRegistry};

use super::{ChapterError, ChapterStage};

const SUMMARY_MARKERS: [&str; 4] = ["当前章节摘要:", "章节摘要:", "摘要:", "本章摘要:"];

pub fn load_recent_chapters(dir: &Path, current: u32, count: usize) -> Vec<String> {
    if current <= 1 || count == 0 {
        return Vec::new();
    }

    let start = current.saturating_sub(count as u32).max(1);
    let mut texts = Vec::new();
    for chapter_number in start..current {
        let path = dir.join(format!("chapter_{}.txt", chapter_number));
        match fs::read_to_string(&path) {
            Ok(text) => texts.push(text),
            Err(err) if err.kind() == io::ErrorKind::NotFound => texts.push(String::new()),
            Err(_) => texts.push(String::new()),
        }
    }
    texts
}

pub fn summarize_recent_chapters<M: LanguageModel + ?Sized>(
    model: &M,
    prompts: &PromptRegistry,
    chapter: &ChapterBlueprintEntry,
    next: Option<&ChapterBlueprintEntry>,
    combined_text: &str,
    chapter_number: u32,
) -> Result<String, ChapterError> {
    let prompt = prompts
        .format(
            "summarize_recent_chapters",
            &summary_arguments(chapter, next, combined_text, chapter_number),
        )
        .map_err(|source| ChapterError::Prompt {
            stage: ChapterStage::Summary,
            source,
        })?;

    model.invoke(&prompt).map_err(|source| ChapterError::Model {
        stage: ChapterStage::Summary,
        source,
    })
}

pub fn extract_summary(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }

    for marker in SUMMARY_MARKERS {
        if let Some(index) = text.find(marker) {
            let (_, rest) = text.split_at(index + marker.len());
            return rest.trim().to_string();
        }
    }

    text.trim().to_string()
}

fn summary_arguments(
    chapter: &ChapterBlueprintEntry,
    next: Option<&ChapterBlueprintEntry>,
    combined_text: &str,
    chapter_number: u32,
) -> PromptArguments {
    let mut args = PromptArguments::new();
    args.insert("combined_text".into(), combined_text.to_string());
    args.insert("novel_number".into(), chapter_number.to_string());
    args.insert("chapter_title".into(), chapter.chapter_title.clone());
    args.insert("chapter_role".into(), chapter.chapter_role.clone());
    args.insert("chapter_purpose".into(), chapter.chapter_purpose.clone());
    args.insert("suspense_level".into(), chapter.suspense_level.clone());
    args.insert("foreshadowing".into(), chapter.foreshadowing.clone());
    args.insert("plot_twist_level".into(), chapter.plot_twist_level.clone());
    args.insert("chapter_summary".into(), chapter.chapter_summary.clone());

    let default_next_number = chapter_number.saturating_add(1);
    let (
        next_title,
        next_role,
        next_purpose,
        next_summary,
        next_suspense,
        next_foreshadowing,
        next_twist,
    ) = next
        .map(|entry| {
            (
                entry.chapter_title.clone(),
                entry.chapter_role.clone(),
                entry.chapter_purpose.clone(),
                entry.chapter_summary.clone(),
                entry.suspense_level.clone(),
                entry.foreshadowing.clone(),
                entry.plot_twist_level.clone(),
            )
        })
        .unwrap_or_else(|| {
            (
                "（未命名）".to_string(),
                "过渡章节".to_string(),
                "承上启下".to_string(),
                "衔接过渡内容".to_string(),
                "中等".to_string(),
                "无特殊伏笔".to_string(),
                "★☆☆☆☆".to_string(),
            )
        });

    args.insert(
        "next_chapter_number".into(),
        default_next_number.to_string(),
    );
    args.insert("next_chapter_title".into(), next_title);
    args.insert("next_chapter_role".into(), next_role);
    args.insert("next_chapter_purpose".into(), next_purpose);
    args.insert("next_chapter_summary".into(), next_summary);
    args.insert("next_chapter_suspense_level".into(), next_suspense);
    args.insert("next_chapter_foreshadowing".into(), next_foreshadowing);
    args.insert("next_chapter_plot_twist_level".into(), next_twist);

    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_summary_by_marker() {
        let response = "一些内容\n当前章节摘要: 这是摘要部分";
        assert_eq!(extract_summary(response), "这是摘要部分");
    }

    #[test]
    fn returns_trimmed_response_when_no_marker() {
        let response = "纯文本摘要";
        assert_eq!(extract_summary(response), "纯文本摘要");
    }
}
