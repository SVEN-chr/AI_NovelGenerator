use crate::blueprint::ChapterBlueprintEntry;
use crate::prompts::{PromptArguments, PromptRegistry};

use super::{ChapterError, ChapterPromptRequest, ChapterStage};

pub fn render_first_chapter_prompt(
    prompts: &PromptRegistry,
    chapter: &ChapterBlueprintEntry,
    request: &ChapterPromptRequest<'_>,
    architecture: &str,
) -> Result<String, ChapterError> {
    let mut args = PromptArguments::new();
    args.insert("novel_number".into(), request.novel_number.to_string());
    args.insert("chapter_title".into(), chapter.chapter_title.clone());
    args.insert("chapter_role".into(), chapter.chapter_role.clone());
    args.insert("chapter_purpose".into(), chapter.chapter_purpose.clone());
    args.insert("suspense_level".into(), chapter.suspense_level.clone());
    args.insert("foreshadowing".into(), chapter.foreshadowing.clone());
    args.insert("plot_twist_level".into(), chapter.plot_twist_level.clone());
    args.insert("chapter_summary".into(), chapter.chapter_summary.clone());
    args.insert("word_number".into(), request.word_number.to_string());
    args.insert(
        "characters_involved".into(),
        request.characters_involved.clone(),
    );
    args.insert("key_items".into(), request.key_items.clone());
    args.insert("scene_location".into(), request.scene_location.clone());
    args.insert("time_constraint".into(), request.time_constraint.clone());
    args.insert("user_guidance".into(), request.user_guidance.clone());
    args.insert("novel_setting".into(), architecture.to_string());

    prompts
        .format("first_chapter_draft", &args)
        .map_err(|source| ChapterError::Prompt {
            stage: ChapterStage::Prompt,
            source,
        })
}

#[allow(clippy::too_many_arguments)]
pub fn render_next_chapter_prompt(
    prompts: &PromptRegistry,
    chapter: &ChapterBlueprintEntry,
    next: Option<&ChapterBlueprintEntry>,
    request: &ChapterPromptRequest<'_>,
    global_summary: &str,
    previous_excerpt: &str,
    character_state: &str,
    short_summary: &str,
    filtered_context: &str,
) -> Result<String, ChapterError> {
    let mut args = PromptArguments::new();
    args.insert(
        "user_guidance".into(),
        user_guidance_value(&request.user_guidance),
    );
    args.insert("global_summary".into(), global_summary.to_string());
    args.insert(
        "previous_chapter_excerpt".into(),
        previous_excerpt.to_string(),
    );
    args.insert("character_state".into(), character_state.to_string());
    args.insert("short_summary".into(), short_summary.to_string());
    args.insert("novel_number".into(), request.novel_number.to_string());
    args.insert("chapter_title".into(), chapter.chapter_title.clone());
    args.insert("chapter_role".into(), chapter.chapter_role.clone());
    args.insert("chapter_purpose".into(), chapter.chapter_purpose.clone());
    args.insert("suspense_level".into(), chapter.suspense_level.clone());
    args.insert("foreshadowing".into(), chapter.foreshadowing.clone());
    args.insert("plot_twist_level".into(), chapter.plot_twist_level.clone());
    args.insert("chapter_summary".into(), chapter.chapter_summary.clone());
    args.insert("word_number".into(), request.word_number.to_string());
    args.insert(
        "characters_involved".into(),
        request.characters_involved.clone(),
    );
    args.insert("key_items".into(), request.key_items.clone());
    args.insert("scene_location".into(), request.scene_location.clone());
    args.insert("time_constraint".into(), request.time_constraint.clone());

    let default_next_number = request.novel_number.saturating_add(1);
    let (
        next_title,
        next_role,
        next_purpose,
        next_suspense,
        next_foreshadowing,
        next_twist,
        next_summary,
    ) = next
        .map(|entry| {
            (
                entry.chapter_title.clone(),
                entry.chapter_role.clone(),
                entry.chapter_purpose.clone(),
                entry.suspense_level.clone(),
                entry.foreshadowing.clone(),
                entry.plot_twist_level.clone(),
                entry.chapter_summary.clone(),
            )
        })
        .unwrap_or_else(|| {
            (
                "（未命名）".to_string(),
                "过渡章节".to_string(),
                "承上启下".to_string(),
                "中等".to_string(),
                "无特殊伏笔".to_string(),
                "★☆☆☆☆".to_string(),
                "衔接过渡内容".to_string(),
            )
        });

    args.insert(
        "next_chapter_number".into(),
        default_next_number.to_string(),
    );
    args.insert("next_chapter_title".into(), next_title);
    args.insert("next_chapter_role".into(), next_role);
    args.insert("next_chapter_purpose".into(), next_purpose);
    args.insert("next_chapter_suspense_level".into(), next_suspense);
    args.insert("next_chapter_foreshadowing".into(), next_foreshadowing);
    args.insert("next_chapter_plot_twist_level".into(), next_twist);
    args.insert("next_chapter_summary".into(), next_summary);
    args.insert("filtered_context".into(), filtered_context.to_string());

    prompts
        .format("next_chapter_draft", &args)
        .map_err(|source| ChapterError::Prompt {
            stage: ChapterStage::Prompt,
            source,
        })
}

fn user_guidance_value(value: &str) -> String {
    if value.trim().is_empty() {
        "无特殊指导".to_string()
    } else {
        value.to_string()
    }
}
