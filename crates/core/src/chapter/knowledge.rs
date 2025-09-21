use regex::Regex;
use std::sync::OnceLock;

use crate::architecture::LanguageModel;
use crate::blueprint::ChapterBlueprintEntry;
use crate::prompts::{PromptArguments, PromptRegistry};

use super::{
    truncate_to_owned, ChapterError, ChapterPromptRequest, ChapterStage, KNOWLEDGE_FALLBACK,
    KNOWLEDGE_SNIPPET_MAX_CHARS,
};

pub fn apply_content_rules(contexts: &[String], chapter_number: u32) -> Vec<String> {
    contexts
        .iter()
        .map(|text| annotate_context(text, chapter_number))
        .collect()
}

pub fn filter_knowledge_contexts<M: LanguageModel + ?Sized>(
    model: &M,
    prompts: &PromptRegistry,
    chapter: &ChapterBlueprintEntry,
    request: &ChapterPromptRequest<'_>,
    contexts: &[String],
) -> Result<String, ChapterError> {
    let processed = apply_knowledge_rules(contexts, request.novel_number);
    let formatted = format_processed_contexts(&processed);

    let mut args = PromptArguments::new();
    args.insert("retrieved_texts".into(), formatted);
    args.insert("chapter_info".into(), format_chapter_info(chapter, request));

    let prompt = prompts
        .format("knowledge_filter", &args)
        .map_err(|source| ChapterError::Prompt {
            stage: ChapterStage::KnowledgeFilter,
            source,
        })?;

    let response = model
        .invoke(&prompt)
        .map_err(|source| ChapterError::Model {
            stage: ChapterStage::KnowledgeFilter,
            source,
        })?;

    let trimmed = response.trim();
    if trimmed.is_empty() {
        Ok(KNOWLEDGE_FALLBACK.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn annotate_context(text: &str, chapter_number: u32) -> String {
    let base_text = text.trim();
    if base_text.is_empty() {
        return String::new();
    }

    if contains_chapter_reference(base_text) {
        let recent = extract_recent_chapter(base_text);
        let distance = chapter_number.saturating_sub(recent);

        if distance <= 2 {
            let preview = preview_with_limit(base_text, 120);
            return format!("[SKIP] 跳过近章内容：{preview}");
        } else if (3..=5).contains(&distance) {
            return format!("[MOD40%] {base_text}（需修改≥40%）");
        } else {
            return format!("[OK] {base_text}（可引用核心）");
        }
    }

    format!("[PRIOR] {base_text}（优先使用）")
}

fn apply_knowledge_rules(contexts: &[String], chapter_number: u32) -> Vec<String> {
    contexts
        .iter()
        .map(|text| {
            let trimmed = text.trim();
            if trimmed.contains('第') && trimmed.contains('章') {
                let recent = extract_recent_chapter(trimmed);
                let distance = chapter_number.saturating_sub(recent);
                if distance <= 3 {
                    let preview = preview_with_limit(trimmed, 50);
                    return format!("[历史章节限制] 跳过近期内容: {preview}");
                }
                format!("[历史参考] {trimmed} (需进行30%以上改写)")
            } else {
                format!("[外部知识] {trimmed}")
            }
        })
        .collect()
}

fn format_processed_contexts(processed: &[String]) -> String {
    if processed.is_empty() {
        return "（无检索结果）".to_string();
    }

    processed
        .iter()
        .enumerate()
        .map(|(idx, text)| {
            let mut snippet = text.trim().to_string();
            if snippet.chars().count() > KNOWLEDGE_SNIPPET_MAX_CHARS {
                snippet = truncate_to_owned(&snippet, KNOWLEDGE_SNIPPET_MAX_CHARS);
                snippet.push_str("...");
            }
            format!("[预处理结果{}]\n{}", idx + 1, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_chapter_info(
    chapter: &ChapterBlueprintEntry,
    request: &ChapterPromptRequest<'_>,
) -> String {
    format!(
        "当前章节定位：{}\n核心目标：{}\n关键要素：{} | {} | {}",
        chapter.chapter_role,
        chapter.chapter_purpose,
        request.characters_involved,
        request.key_items,
        request.scene_location,
    )
}

fn contains_chapter_reference(text: &str) -> bool {
    static CHAPTER_RE: OnceLock<Regex> = OnceLock::new();
    let regex = CHAPTER_RE.get_or_init(|| Regex::new(r"第\s*\d+\s*章|chapter_\d+").unwrap());
    regex.is_match(text)
}

fn extract_recent_chapter(text: &str) -> u32 {
    static DIGIT_RE: OnceLock<Regex> = OnceLock::new();
    let regex = DIGIT_RE.get_or_init(|| Regex::new(r"\d+").unwrap());
    regex
        .find_iter(text)
        .filter_map(|m| m.as_str().parse::<u32>().ok())
        .max()
        .unwrap_or(0)
}

fn preview_with_limit(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let mut snippet = truncate_to_owned(text, max_chars);
        snippet.push_str("...");
        snippet
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_rules_mark_recent_chapters() {
        let contexts = vec!["chapter_10 发生冲突".to_string(), "全新设定".to_string()];
        let annotated = apply_content_rules(&contexts, 11);
        assert!(annotated[0].starts_with("[SKIP]"));
        assert!(annotated[1].starts_with("[PRIOR]"));
    }

    #[test]
    fn knowledge_rules_flag_recent_history() {
        let contexts = vec!["第11章 冲突升级".to_string(), "外部知识".to_string()];
        let processed = apply_knowledge_rules(&contexts, 12);
        assert!(processed[0].contains("[历史章节限制]"));
        assert!(processed[1].contains("[外部知识]"));
    }
}
