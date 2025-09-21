use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use log::debug;
use reqwest::blocking::{Client, RequestBuilder, Response};
use reqwest::Method;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use novel_core::chapter::{KnowledgeBase, KnowledgeBaseError};
use novel_core::embedding::EmbeddingModel;
use novel_core::logging::{LogLevel, LogRecord, LogSink};
use novel_core::vectorstore::{split_text_segments, DEFAULT_SEGMENT_CHAR_LIMIT};

use crate::error::AdapterError;
use crate::retry::{call_with_retry, RetryConfig};

const VECTORSTORE_DIR_NAME: &str = "vectorstore";
const METADATA_FILE_NAME: &str = "metadata.json";
const MAX_BATCH_SIZE: usize = 32;
const MAX_CONTEXT_CHARS: usize = 2_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub base_url: String,
    pub collection_name: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

impl VectorStoreConfig {
    fn normalized_base_url(&self) -> String {
        let trimmed = self.base_url.trim();
        if trimmed.ends_with('/') {
            trimmed[..trimmed.len() - 1].to_string()
        } else {
            trimmed.to_string()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct VectorStoreMetadata {
    config: VectorStoreConfig,
    vector_size: usize,
}

pub struct QdrantVectorStore {
    http: QdrantHttpClient,
    embedding: Arc<dyn EmbeddingModel>,
    metadata_path: PathBuf,
    metadata: VectorStoreMetadata,
}

impl QdrantVectorStore {
    fn new(
        http: QdrantHttpClient,
        embedding: Arc<dyn EmbeddingModel>,
        metadata_path: PathBuf,
        metadata: VectorStoreMetadata,
    ) -> Self {
        Self {
            http,
            embedding,
            metadata_path,
            metadata,
        }
    }

    fn upsert_segments(
        &self,
        sink: &dyn LogSink,
        chapter: Option<u32>,
        segments: &[String],
    ) -> Result<usize, AdapterError> {
        if segments.is_empty() {
            sink.log(LogRecord::new(
                LogLevel::Warn,
                "未找到可写入向量库的文本片段，已跳过。",
            ));
            return Ok(0);
        }

        let vectors = self
            .embedding
            .embed_documents(segments)
            .map_err(AdapterError::from)?;
        if vectors.is_empty() {
            sink.log(LogRecord::new(
                LogLevel::Warn,
                "向量模型返回空结果，向量库更新已跳过。",
            ));
            return Ok(0);
        }

        let dimension = vectors.first().map(|v| v.len()).unwrap_or(0);
        if dimension == 0 {
            return Err(AdapterError::VectorStore(
                "embedding model returned zero-dimension vectors".to_string(),
            ));
        }

        if dimension != self.metadata.vector_size {
            return Err(AdapterError::VectorStore(format!(
                "embedding dimension changed ({} -> {}), 请重新初始化向量库",
                self.metadata.vector_size, dimension
            )));
        }

        let mut total = 0usize;
        for (batch_index, chunk) in segments
            .iter()
            .cloned()
            .zip(vectors.into_iter())
            .collect::<Vec<(String, Vec<f32>)>>()
            .chunks(MAX_BATCH_SIZE)
            .enumerate()
        {
            let points: Vec<QdrantPoint> = chunk
                .iter()
                .enumerate()
                .map(|(offset, (text, vector))| {
                    QdrantPoint::new(text, vector.clone(), chapter, total + offset)
                })
                .collect();

            self.http.upsert_points(&points)?;
            total += chunk.len();
            debug!(
                "向量库批量写入完成：batch={} size={}",
                batch_index,
                chunk.len()
            );
        }

        sink.log(LogRecord::new(
            LogLevel::Info,
            format!("向量库写入完成，共 {} 条片段。", total),
        ));

        Ok(total)
    }

    fn similarity_query(&self, query: &str, limit: usize) -> Result<String, AdapterError> {
        if query.trim().is_empty() {
            return Ok(String::new());
        }

        let vector = self
            .embedding
            .embed_query(query)
            .map_err(AdapterError::from)?;
        let results = self.http.search(&vector, limit.max(1))?;
        Ok(truncate_combined_text(results, MAX_CONTEXT_CHARS))
    }

    fn record_metadata(&self, metadata: &VectorStoreMetadata) -> Result<(), AdapterError> {
        let json = serde_json::to_vec_pretty(metadata)?;
        fs::write(&self.metadata_path, json)
            .map_err(|source| AdapterError::io(&self.metadata_path, source))
    }
}

impl KnowledgeBase for QdrantVectorStore {
    fn count(&self) -> Result<Option<usize>, KnowledgeBaseError> {
        self.http
            .count()
            .map(Some)
            .map_err(|err| KnowledgeBaseError::new(err))
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<String>, KnowledgeBaseError> {
        let vector = self
            .embedding
            .embed_query(query)
            .map_err(|err| KnowledgeBaseError::new(AdapterError::from(err)))?;
        self.http
            .search(&vector, limit.max(1))
            .map_err(KnowledgeBaseError::new)
    }
}

pub fn get_vectorstore_dir(workspace: &Path) -> PathBuf {
    workspace.join(VECTORSTORE_DIR_NAME)
}

fn metadata_path(workspace: &Path) -> PathBuf {
    get_vectorstore_dir(workspace).join(METADATA_FILE_NAME)
}

pub fn load_vector_store(
    sink: &dyn LogSink,
    embedding: Arc<dyn EmbeddingModel>,
    workspace: &Path,
) -> Result<Option<QdrantVectorStore>, AdapterError> {
    let path = metadata_path(workspace);
    if !path.exists() {
        sink.log(LogRecord::new(
            LogLevel::Info,
            "未找到向量库元数据，跳过加载。",
        ));
        return Ok(None);
    }

    let data = fs::read_to_string(&path).map_err(|source| AdapterError::io(&path, source))?;
    let metadata: VectorStoreMetadata = serde_json::from_str(&data)?;
    let http = QdrantHttpClient::new(&metadata.config)?;

    Ok(Some(QdrantVectorStore::new(
        http, embedding, path, metadata,
    )))
}

pub fn init_vector_store(
    sink: &dyn LogSink,
    embedding: Arc<dyn EmbeddingModel>,
    workspace: &Path,
    config: VectorStoreConfig,
    segments: &[String],
) -> Result<(QdrantVectorStore, usize), AdapterError> {
    if segments.is_empty() {
        return Err(AdapterError::VectorStore(
            "知识库初始化失败：待插入文本为空".to_string(),
        ));
    }

    let dir = get_vectorstore_dir(workspace);
    fs::create_dir_all(&dir).map_err(|source| AdapterError::io(&dir, source))?;
    let path = dir.join(METADATA_FILE_NAME);

    let vectors = embedding
        .embed_documents(segments)
        .map_err(AdapterError::from)?;
    let vector_size = vectors.first().map(|v| v.len()).unwrap_or(0);
    if vector_size == 0 {
        return Err(AdapterError::VectorStore(
            "embedding model returned zero-dimension vectors".to_string(),
        ));
    }

    let metadata = VectorStoreMetadata {
        config: config.clone(),
        vector_size,
    };

    let http = QdrantHttpClient::new(&config)?;
    http.recreate_collection(vector_size)?;

    let store = QdrantVectorStore::new(http, embedding, path.clone(), metadata.clone());
    store.record_metadata(&metadata)?;

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("开始初始化向量库（共 {} 个片段）", segments.len()),
    ));

    let inserted = store.upsert_segments(sink, None, segments)?;
    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("向量库初始化完成，共写入 {} 条片段。", inserted),
    ));

    Ok((store, inserted))
}

pub fn update_vector_store(
    sink: &dyn LogSink,
    store: &QdrantVectorStore,
    chapter_number: u32,
    chapter_text: &str,
) -> Result<usize, AdapterError> {
    let segments = split_text_segments(chapter_text, DEFAULT_SEGMENT_CHAR_LIMIT);
    if segments.is_empty() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            format!("第{}章定稿内容为空，向量库更新跳过。", chapter_number),
        ));
        return Ok(0);
    }

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!(
            "更新向量库：第{}章，共 {} 个片段。",
            chapter_number,
            segments.len()
        ),
    ));

    store.upsert_segments(sink, Some(chapter_number), &segments)
}

pub fn import_knowledge_file(
    sink: &dyn LogSink,
    embedding: Arc<dyn EmbeddingModel>,
    workspace: &Path,
    config: VectorStoreConfig,
    file_path: &Path,
) -> Result<usize, AdapterError> {
    if !file_path.exists() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            format!("知识库文件不存在：{}", file_path.display()),
        ));
        return Ok(0);
    }

    let content =
        fs::read_to_string(file_path).map_err(|source| AdapterError::io(file_path, source))?;
    if content.trim().is_empty() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            "知识库文件内容为空，已跳过导入。",
        ));
        return Ok(0);
    }

    let segments = split_text_segments(&content, DEFAULT_SEGMENT_CHAR_LIMIT);
    if segments.is_empty() {
        sink.log(LogRecord::new(
            LogLevel::Warn,
            "知识库文本切分后为空，已跳过导入。",
        ));
        return Ok(0);
    }

    sink.log(LogRecord::new(
        LogLevel::Info,
        format!("准备导入知识库文件，共 {} 个片段。", segments.len()),
    ));

    match load_vector_store(sink, embedding.clone(), workspace)? {
        Some(store) => store.upsert_segments(sink, None, &segments),
        None => {
            let (_store, inserted) =
                init_vector_store(sink, embedding, workspace, config, &segments)?;
            Ok(inserted)
        }
    }
}

pub fn similarity_search(
    store: &QdrantVectorStore,
    query: &str,
    k: usize,
) -> Result<String, AdapterError> {
    store.similarity_query(query, k)
}

#[derive(Serialize)]
struct QdrantPoint {
    id: String,
    vector: Vec<f32>,
    payload: serde_json::Value,
}

impl QdrantPoint {
    fn new(text: &str, vector: Vec<f32>, chapter: Option<u32>, segment_index: usize) -> Self {
        let mut payload = serde_json::Map::new();
        payload.insert("text".into(), serde_json::Value::String(text.to_string()));
        payload.insert(
            "segment".into(),
            serde_json::Value::Number(segment_index.into()),
        );
        if let Some(chapter) = chapter {
            payload.insert("chapter".into(), serde_json::Value::Number(chapter.into()));
        }

        Self {
            id: Uuid::new_v4().to_string(),
            vector,
            payload: serde_json::Value::Object(payload),
        }
    }
}

struct QdrantHttpClient {
    client: Client,
    base_url: String,
    collection_name: String,
    api_key: Option<String>,
    retry: RetryConfig,
}

impl QdrantHttpClient {
    fn new(config: &VectorStoreConfig) -> Result<Self, AdapterError> {
        if config.base_url.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "向量库 base_url 不能为空".to_string(),
            ));
        }
        if config.collection_name.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "向量库 collection_name 不能为空".to_string(),
            ));
        }

        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
            base_url: config.normalized_base_url(),
            collection_name: config.collection_name.trim().to_string(),
            api_key: config.api_key.clone(),
            retry: RetryConfig::default(),
        })
    }

    fn recreate_collection(&self, vector_size: usize) -> Result<(), AdapterError> {
        let path = format!("/collections/{}", self.collection_name);
        call_with_retry(|| self.delete_collection(&path), &self.retry)?;
        call_with_retry(|| self.create_collection(&path, vector_size), &self.retry)
    }

    fn upsert_points(&self, points: &[QdrantPoint]) -> Result<(), AdapterError> {
        let path = format!("/collections/{}/points?wait=true", self.collection_name);
        call_with_retry(
            || {
                let response = self
                    .request(Method::PUT, &path)?
                    .json(&serde_json::json!({"points": points}))
                    .send()?;
                Self::ensure_success(response)
            },
            &self.retry,
        )
    }

    fn search(&self, vector: &[f32], limit: usize) -> Result<Vec<String>, AdapterError> {
        let path = format!("/collections/{}/points/search", self.collection_name);
        call_with_retry(
            || {
                let response = self
                    .request(Method::POST, &path)?
                    .json(&serde_json::json!({
                        "vector": vector,
                        "limit": limit,
                        "with_payload": true,
                    }))
                    .send()?;
                let value = Self::parse_json(response)?;
                Ok(parse_search_payload(value))
            },
            &self.retry,
        )
    }

    fn count(&self) -> Result<usize, AdapterError> {
        let path = format!(
            "/collections/{}/points/count?exact=true",
            self.collection_name
        );
        call_with_retry(
            || {
                let response = self.request(Method::POST, &path)?.send()?;
                let value = Self::parse_json(response)?;
                Ok(value
                    .get("result")
                    .and_then(|r| r.get("count"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize)
            },
            &self.retry,
        )
    }

    fn delete_collection(&self, path: &str) -> Result<(), AdapterError> {
        let response = self.request(Method::DELETE, path)?.send()?;
        if response.status().is_success() || response.status().as_u16() == 404 {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(AdapterError::HttpStatus { status, body })
        }
    }

    fn create_collection(&self, path: &str, vector_size: usize) -> Result<(), AdapterError> {
        let payload = serde_json::json!({
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        });
        let response = self.request(Method::PUT, path)?.json(&payload).send()?;
        Self::ensure_success(response)
    }

    fn request(&self, method: Method, path: &str) -> Result<RequestBuilder, AdapterError> {
        let url = format!("{}{}", self.base_url, path);
        let mut builder = self.client.request(method, &url);
        if let Some(key) = &self.api_key {
            if !key.trim().is_empty() {
                builder = builder.header("api-key", key);
            }
        }
        Ok(builder)
    }

    fn ensure_success(response: Response) -> Result<(), AdapterError> {
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(AdapterError::HttpStatus { status, body })
        }
    }

    fn parse_json(response: Response) -> Result<serde_json::Value, AdapterError> {
        if response.status().is_success() {
            response
                .json::<serde_json::Value>()
                .map_err(AdapterError::from)
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(AdapterError::HttpStatus { status, body })
        }
    }
}

fn truncate_combined_text(chunks: Vec<String>, max_chars: usize) -> String {
    if chunks.is_empty() || max_chars == 0 {
        return String::new();
    }

    let mut combined = String::new();
    for chunk in chunks {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(trimmed);
        if combined.chars().count() >= max_chars {
            let truncated: String = combined.chars().take(max_chars).collect();
            return truncated;
        }
    }

    if combined.chars().count() > max_chars {
        combined.chars().take(max_chars).collect()
    } else {
        combined
    }
}

fn parse_search_payload(value: serde_json::Value) -> Vec<String> {
    value
        .get("result")
        .and_then(|result| result.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.get("payload")
                        .and_then(|payload| payload.get("text"))
                        .and_then(|text| text.as_str())
                        .map(|text| text.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}
