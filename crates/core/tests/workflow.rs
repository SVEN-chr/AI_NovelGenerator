use std::collections::VecDeque;
use std::fs;
use std::io;
use std::sync::Mutex;

use novel_core::architecture::ARCHITECTURE_FILE_NAME;
use novel_core::{
    ArchitectureRequest, ArchitectureService, ChapterBlueprintRequest, ChapterBlueprintService,
    ChapterFinalizer, ChapterPromptRequest, ChapterService, FinalizeChapterRequest, KnowledgeBase,
    KnowledgeBaseError, LanguageModel, LanguageModelError, PromptRegistry, VecLogSink,
    BLUEPRINT_FILE_NAME,
};
use tempfile::tempdir;

struct MockLanguageModel {
    responses: Mutex<VecDeque<String>>,
}

impl MockLanguageModel {
    fn new<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            responses: Mutex::new(responses.into_iter().map(Into::into).collect()),
        }
    }

    fn assert_empty(&self) {
        let guard = self.responses.lock().expect("mock mutex poisoned");
        assert!(
            guard.is_empty(),
            "expected all mock responses to be consumed"
        );
    }
}

impl LanguageModel for MockLanguageModel {
    fn invoke(&self, _prompt: &str) -> Result<String, LanguageModelError> {
        let mut guard = self.responses.lock().expect("mock mutex poisoned");
        guard.pop_front().ok_or_else(|| {
            LanguageModelError::new(io::Error::new(
                io::ErrorKind::Other,
                "mock language model has no remaining responses",
            ))
        })
    }
}

struct NoKnowledge;

impl KnowledgeBase for NoKnowledge {
    fn search(&self, _query: &str, _limit: usize) -> Result<Vec<String>, KnowledgeBaseError> {
        Ok(Vec::new())
    }
}

#[test]
fn full_generation_pipeline_produces_expected_artifacts() -> Result<(), Box<dyn std::error::Error>>
{
    const BLUEPRINT_TEXT: &str = "\
第1章 - [序章]\n\
本章定位： [开端]\n\
核心作用： [介绍背景]\n\
悬念密度： [中]\n\
伏笔操作： [埋下伏笔]\n\
认知颠覆： [低]\n\
本章简述： [开篇描写]\n\n\
第2章 - [冲突]\n\
本章定位： [发展]\n\
核心作用： [推进情节]\n\
悬念密度： [高]\n\
伏笔操作： [回收伏笔]\n\
认知颠覆： [中]\n\
本章简述： [矛盾升级]\n";

    let temp = tempdir()?;
    let workspace = temp.path();

    let prompts = PromptRegistry::new()?;
    let sink = VecLogSink::new();

    let mock = MockLanguageModel::new([
        "核心种子描写".to_string(),
        "角色动力学设定".to_string(),
        "初始角色状态".to_string(),
        "世界观补充".to_string(),
        "三幕式情节".to_string(),
        BLUEPRINT_TEXT.to_string(),
        "第一章正文内容".to_string(),
        "新的摘要".to_string(),
        "新的角色状态".to_string(),
    ]);

    let architecture_request = ArchitectureRequest {
        topic: "测试主题".into(),
        genre: "测试类型".into(),
        number_of_chapters: 2,
        word_number: 800,
        user_guidance: "补充说明".into(),
    };

    let architecture_service = ArchitectureService::new(&prompts, &sink);
    architecture_service.generate(&mock, workspace, &architecture_request)?;

    assert!(workspace.join(ARCHITECTURE_FILE_NAME).exists());
    assert!(workspace.join("character_state.txt").exists());

    let blueprint_service = ChapterBlueprintService::new(&prompts, &sink);
    let blueprint_request = ChapterBlueprintRequest::new(2, "", 4096);
    let blueprint = blueprint_service.generate(&mock, workspace, &blueprint_request)?;
    assert_eq!(blueprint.len(), 2);

    let mut chapter_request = ChapterPromptRequest::new(workspace, &blueprint, 1, 800);
    chapter_request.user_guidance = "章节指导".into();

    let chapter_service = ChapterService::new(&prompts, &sink);
    let draft = chapter_service.generate_chapter_draft::<_, NoKnowledge>(
        &mock,
        None,
        &chapter_request,
        None,
    )?;
    assert!(draft.path.exists());
    assert!(draft.content.contains("第一章正文内容"));

    let finalizer = ChapterFinalizer::new(&prompts, &sink);
    let finalize_request = FinalizeChapterRequest {
        output_dir: workspace.to_path_buf(),
        chapter_number: 1,
    };
    let finalize_result = finalizer.finalize_chapter(&mock, None, &finalize_request)?;

    let summary = fs::read_to_string(&finalize_result.summary_path)?;
    assert!(summary.contains("新的摘要"));

    let character_state = fs::read_to_string(&finalize_result.character_state_path)?;
    assert!(character_state.contains("新的角色状态"));

    assert!(workspace.join(BLUEPRINT_FILE_NAME).exists());

    mock.assert_empty();

    Ok(())
}
