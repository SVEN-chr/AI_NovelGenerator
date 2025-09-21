use crate::architecture::LanguageModel;
use crate::blueprint::ChapterBlueprintEntry;
use crate::prompts::{PromptArguments, PromptRegistry};

use super::{ChapterError, ChapterPromptRequest, ChapterStage};

pub fn generate_keyword_groups<M: LanguageModel + ?Sized>(
    model: &M,
    prompts: &PromptRegistry,
    chapter: &ChapterBlueprintEntry,
    request: &ChapterPromptRequest<'_>,
    summary: &str,
) -> Result<Vec<String>, ChapterError> {
    let prompt = prompts
        .format(
            "knowledge_search",
            &keyword_arguments(chapter, request, summary),
        )
        .map_err(|source| ChapterError::Prompt {
            stage: ChapterStage::KeywordGeneration,
            source,
        })?;

    let response = model
        .invoke(&prompt)
        .map_err(|source| ChapterError::Model {
            stage: ChapterStage::KeywordGeneration,
            source,
        })?;

    Ok(parse_keyword_groups(&response))
}

pub fn parse_keyword_groups(response: &str) -> Vec<String> {
    response
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.contains('·') {
                Some(trimmed.replace('·', " "))
            } else {
                None
            }
        })
        .take(5)
        .collect()
}

fn keyword_arguments(
    chapter: &ChapterBlueprintEntry,
    request: &ChapterPromptRequest<'_>,
    summary: &str,
) -> PromptArguments {
    let mut args = PromptArguments::new();
    args.insert("chapter_number".into(), request.novel_number.to_string());
    args.insert("chapter_title".into(), chapter.chapter_title.clone());
    args.insert(
        "characters_involved".into(),
        request.characters_involved.clone(),
    );
    args.insert("key_items".into(), request.key_items.clone());
    args.insert("scene_location".into(), request.scene_location.clone());
    args.insert("chapter_role".into(), chapter.chapter_role.clone());
    args.insert("chapter_purpose".into(), chapter.chapter_purpose.clone());
    args.insert("foreshadowing".into(), chapter.foreshadowing.clone());
    args.insert("short_summary".into(), summary.to_string());
    args.insert("user_guidance".into(), request.user_guidance.clone());
    args.insert("time_constraint".into(), request.time_constraint.clone());
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_keyword_lines() {
        let input = "科技公司·数据泄露\n无效行\n地下实验室·基因编辑·禁忌实验";
        let groups = parse_keyword_groups(input);
        assert_eq!(
            groups,
            vec!["科技公司 数据泄露", "地下实验室 基因编辑 禁忌实验"]
        );
    }
}
