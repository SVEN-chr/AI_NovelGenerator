use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::embedding::{EmbeddingModel, EmbeddingModelError};
use crate::logging::{LogLevel, LogRecord, LogSink};

const VECTOR_STORE_DIR_NAME: &str = "vectorstore";
const VECTOR_STORE_FILE_NAME: &str = "chapters.jsonl";
const VECTOR_SEGMENT_MAX_CHARS: usize = 500;

#[derive(Debug, Error)]
pub enum VectorStoreError {
    #[error("embedding generation failed: {0}")]
    Embedding(#[from] EmbeddingModelError),
    #[error("failed to create vector store directory `{path}`: {source}")]
    CreateDir { path: PathBuf, source: io::Error },
    #[error("failed to open vector store file `{path}`: {source}")]
    OpenFile { path: PathBuf, source: io::Error },
    #[error("failed to write vector store file `{path}`: {source}")]
    WriteFile { path: PathBuf, source: io::Error },
    #[error("failed to serialize vector store record: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Serialize)]
struct VectorStoreRecord {
    chapter: u32,
    segment: usize,
    text: String,
    embedding: Vec<f32>,
}

pub fn update_vector_store(
    sink: &dyn LogSink,
    embedding: &dyn EmbeddingModel,
    output_dir: &Path,
    chapter_number: u32,
    chapter_text: &str,
) -> Result<usize, VectorStoreError> {
    let segments = split_text_for_vectorstore(chapter_text, VECTOR_SEGMENT_MAX_CHARS);
    if segments.is_empty() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            "向量库更新跳过：章节文本为空或无法拆分有效片段。".to_string(),
        ));
        return Ok(0);
    }

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("准备写入向量库，共 {} 个片段。", segments.len()),
    ));

    let embeddings = embedding.embed_documents(&segments)?;
    if embeddings.is_empty() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            "向量模型返回空结果，跳过向量库更新。".to_string(),
        ));
        return Ok(0);
    }

    let zipped_len = segments.len().min(embeddings.len());
    if zipped_len < segments.len() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            format!(
                "向量数量({})少于片段数量({})，仅写入匹配的部分。",
                embeddings.len(),
                segments.len()
            ),
        ));
    }

    let store_dir = output_dir.join(VECTOR_STORE_DIR_NAME);
    fs::create_dir_all(&store_dir).map_err(|source| VectorStoreError::CreateDir {
        path: store_dir.clone(),
        source,
    })?;

    let store_path = store_dir.join(VECTOR_STORE_FILE_NAME);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&store_path)
        .map_err(|source| VectorStoreError::OpenFile {
            path: store_path.clone(),
            source,
        })?;

    for (index, (text, vector)) in segments.iter().zip(embeddings.into_iter()).enumerate() {
        let record = VectorStoreRecord {
            chapter: chapter_number,
            segment: index,
            text: text.trim().to_string(),
            embedding: vector,
        };

        let line = serde_json::to_string(&record)?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|source| VectorStoreError::WriteFile {
                path: store_path.clone(),
                source,
            })?;
    }

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("向量库更新完成：写入 {} 条片段。", zipped_len),
    ));

    Ok(zipped_len)
}

pub fn split_text_for_vectorstore(text: &str, max_length: usize) -> Vec<String> {
    if text.trim().is_empty() || max_length == 0 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = String::new();

    for token in text.split_whitespace() {
        if token.is_empty() {
            continue;
        }

        let token_len = token.chars().count();
        if token_len > max_length {
            if !current.is_empty() {
                segments.push(current.clone());
                current.clear();
            }

            let mut buffer = String::new();
            for ch in token.chars() {
                if buffer.chars().count() == max_length {
                    segments.push(buffer.clone());
                    buffer.clear();
                }
                buffer.push(ch);
            }
            if !buffer.is_empty() {
                segments.push(buffer);
            }
            continue;
        }

        let current_len = current.chars().count();
        let required = if current.is_empty() {
            token_len
        } else {
            current_len + 1 + token_len
        };

        if required > max_length && !current.is_empty() {
            segments.push(current.clone());
            current.clear();
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(token);
    }

    if !current.is_empty() {
        segments.push(current);
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_text_handles_empty_input() {
        assert!(split_text_for_vectorstore("", 10).is_empty());
    }

    #[test]
    fn split_text_respects_max_length() {
        let text = "这是第一段。\n这是第二段，长度会超过限制。";
        let segments = split_text_for_vectorstore(text, 10);
        assert!(!segments.is_empty());
        assert!(segments.iter().all(|s| s.chars().count() <= 10));
    }
}
