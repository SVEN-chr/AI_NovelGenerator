use std::time::Duration;

use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::blocking::Client;
use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::Deserialize;

use novel_core::config::{Config, EmbeddingConfig};
use novel_core::embedding::{EmbeddingModel, EmbeddingModelError};

use crate::base_url::ensure_openai_base_url_has_v1;
use crate::error::AdapterError;
use crate::retry::{call_with_retry, RetryConfig};

pub fn create_embedding_adapter(
    config: &Config,
    profile_name: &str,
) -> Result<Box<dyn EmbeddingModel>, AdapterError> {
    let profile = config.get_embedding_profile(profile_name).ok_or_else(|| {
        AdapterError::InvalidConfig(format!("unknown embedding profile `{}`", profile_name))
    })?;
    create_embedding_adapter_from_profile(profile)
}

pub fn create_embedding_adapter_from_profile(
    profile: &EmbeddingConfig,
) -> Result<Box<dyn EmbeddingModel>, AdapterError> {
    let fmt = profile.interface_format.trim().to_lowercase();

    match fmt.as_str() {
        "openai" => Ok(Box::new(OpenAiEmbeddingAdapter::new(
            optional_string(&profile.api_key),
            &profile.base_url,
            &profile.model_name,
        )?)),
        "azure openai" => Ok(Box::new(AzureOpenAiEmbeddingAdapter::new(
            profile.api_key.clone(),
            &profile.base_url,
        )?)),
        "ollama" => Ok(Box::new(OllamaEmbeddingAdapter::new(
            &profile.base_url,
            &profile.model_name,
        )?)),
        "ml studio" => Ok(Box::new(OpenAiEmbeddingAdapter::new(
            optional_string(&profile.api_key),
            &profile.base_url,
            &profile.model_name,
        )?)),
        "gemini" => Ok(Box::new(GeminiEmbeddingAdapter::new(
            profile.api_key.clone(),
            &profile.base_url,
            &profile.model_name,
        )?)),
        "siliconflow" => Ok(Box::new(OpenAiEmbeddingAdapter::new(
            optional_string(&profile.api_key),
            &profile.base_url,
            &profile.model_name,
        )?)),
        other => Err(AdapterError::InvalidConfig(format!(
            "unknown embedding interface_format: {}",
            other
        ))),
    }
}

fn optional_string(value: &str) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

struct OpenAiEmbeddingAdapter {
    client: Client,
    url: String,
    api_key: Option<String>,
    model_name: String,
    retry: RetryConfig,
}

impl OpenAiEmbeddingAdapter {
    fn new(
        api_key: Option<String>,
        base_url: &str,
        model_name: &str,
    ) -> Result<Self, AdapterError> {
        if model_name.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "embedding model_name must not be empty".to_string(),
            ));
        }

        let normalized = ensure_openai_base_url_has_v1(if base_url.trim().is_empty() {
            "https://api.openai.com"
        } else {
            base_url
        });

        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
            url: format!("{}/embeddings", normalized.trim_end_matches('/')),
            api_key,
            model_name: model_name.to_string(),
            retry: RetryConfig::default(),
        })
    }

    fn embed(&self, input: serde_json::Value) -> Result<Vec<Vec<f32>>, AdapterError> {
        let mut request = self.client.post(&self.url).header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                request = request.bearer_auth(key);
            }
        }

        let payload = serde_json::json!({
            "model": self.model_name,
            "input": input,
        });

        let response = request.json(&payload).send()?;
        handle_openai_embedding_response(response)
    }
}

impl EmbeddingModel for OpenAiEmbeddingAdapter {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingModelError> {
        call_with_retry(
            || {
                let inputs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                self.embed(serde_json::Value::from(inputs))
            },
            &self.retry,
        )
        .map_err(EmbeddingModelError::new)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingModelError> {
        let vectors = call_with_retry(
            || self.embed(serde_json::Value::from(vec![text])),
            &self.retry,
        )
        .map_err(EmbeddingModelError::new)?;
        match vectors.into_iter().next() {
            Some(vector) => Ok(vector),
            None => Err(EmbeddingModelError::new(AdapterError::EmptyResponse)),
        }
    }
}

struct AzureOpenAiEmbeddingAdapter {
    client: Client,
    url: String,
    api_key: String,
    retry: RetryConfig,
}

impl AzureOpenAiEmbeddingAdapter {
    fn new(api_key: String, base_url: &str) -> Result<Self, AdapterError> {
        if api_key.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Azure OpenAI embedding api_key must not be empty".to_string(),
            ));
        }

        static AZURE_RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(
                r"^https://([^/]+)/openai/deployments/([^/]+)/embeddings\?api-version=([^/?&]+)",
            )
            .unwrap()
        });

        let base = base_url.trim();
        let captures = AZURE_RE.captures(base).ok_or_else(|| {
            AdapterError::InvalidConfig(
                "Invalid Azure OpenAI embedding base_url. Expected https://<resource>.openai.azure.com/openai/deployments/<deployment>/embeddings?api-version=<version>"
                    .to_string(),
            )
        })?;

        let endpoint = format!("https://{}", &captures[1]);
        let deployment = captures[2].to_string();
        let api_version = captures[3].to_string();

        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
            url: format!(
                "{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
            ),
            api_key,
            retry: RetryConfig::default(),
        })
    }

    fn embed(&self, input: serde_json::Value) -> Result<Vec<Vec<f32>>, AdapterError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "api-key",
            HeaderValue::from_str(&self.api_key).map_err(|err| {
                AdapterError::InvalidConfig(format!("invalid api key header: {}", err))
            })?,
        );

        let payload = serde_json::json!({
            "input": input,
        });

        let response = self
            .client
            .post(&self.url)
            .headers(headers)
            .json(&payload)
            .send()?;
        handle_openai_embedding_response(response)
    }
}

impl EmbeddingModel for AzureOpenAiEmbeddingAdapter {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingModelError> {
        call_with_retry(
            || {
                let inputs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                self.embed(serde_json::Value::from(inputs))
            },
            &self.retry,
        )
        .map_err(EmbeddingModelError::new)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingModelError> {
        let vectors = call_with_retry(
            || self.embed(serde_json::Value::from(vec![text])),
            &self.retry,
        )
        .map_err(EmbeddingModelError::new)?;
        match vectors.into_iter().next() {
            Some(vector) => Ok(vector),
            None => Err(EmbeddingModelError::new(AdapterError::EmptyResponse)),
        }
    }
}

struct OllamaEmbeddingAdapter {
    client: Client,
    url: String,
    model_name: String,
    retry: RetryConfig,
}

impl OllamaEmbeddingAdapter {
    fn new(base_url: &str, model_name: &str) -> Result<Self, AdapterError> {
        if base_url.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Ollama embedding base_url must not be empty".to_string(),
            ));
        }

        if model_name.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Ollama embedding model_name must not be empty".to_string(),
            ));
        }

        let url = normalize_ollama_url(base_url);

        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
            url,
            model_name: model_name.to_string(),
            retry: RetryConfig::default(),
        })
    }

    fn embed_once(&self, text: &str) -> Result<Vec<f32>, AdapterError> {
        let payload = serde_json::json!({
            "model": self.model_name,
            "prompt": text,
        });

        let response = self.client.post(&self.url).json(&payload).send()?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(AdapterError::HttpStatus { status, body });
        }

        let parsed: OllamaEmbeddingResponse = response.json()?;
        parsed
            .embedding
            .map(|values| values.into_iter().map(|v| v as f32).collect())
            .ok_or(AdapterError::EmptyResponse)
    }
}

impl EmbeddingModel for OllamaEmbeddingAdapter {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingModelError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            let vector = call_with_retry(|| self.embed_once(text), &self.retry)
                .map_err(EmbeddingModelError::new)?;
            embeddings.push(vector);
        }
        Ok(embeddings)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingModelError> {
        call_with_retry(|| self.embed_once(text), &self.retry).map_err(EmbeddingModelError::new)
    }
}

struct GeminiEmbeddingAdapter {
    client: Client,
    url: String,
    retry: RetryConfig,
}

impl GeminiEmbeddingAdapter {
    fn new(api_key: String, base_url: &str, model_name: &str) -> Result<Self, AdapterError> {
        if api_key.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Gemini embedding api_key must not be empty".to_string(),
            ));
        }

        if model_name.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Gemini embedding model_name must not be empty".to_string(),
            ));
        }

        let base = if base_url.trim().is_empty() {
            "https://generativelanguage.googleapis.com/v1beta".to_string()
        } else {
            base_url.trim().trim_end_matches('/').to_string()
        };

        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
            url: format!(
                "{base}/models/{model}:embedContent?key={api}",
                model = model_name,
                api = api_key
            ),
            retry: RetryConfig::default(),
        })
    }

    fn embed_once(&self, text: &str) -> Result<Vec<f32>, AdapterError> {
        let payload = serde_json::json!({
            "content": {
                "parts": [ { "text": text } ]
            }
        });

        let response = self.client.post(&self.url).json(&payload).send()?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(AdapterError::HttpStatus { status, body });
        }

        let parsed: GeminiEmbeddingResponse = response.json()?;
        parsed
            .embedding
            .map(|values| values.into_iter().map(|v| v as f32).collect())
            .ok_or(AdapterError::EmptyResponse)
    }
}

impl EmbeddingModel for GeminiEmbeddingAdapter {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingModelError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            let vector = call_with_retry(|| self.embed_once(text), &self.retry)
                .map_err(EmbeddingModelError::new)?;
            embeddings.push(vector);
        }
        Ok(embeddings)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingModelError> {
        call_with_retry(|| self.embed_once(text), &self.retry).map_err(EmbeddingModelError::new)
    }
}

fn normalize_ollama_url(base_url: &str) -> String {
    let mut url = base_url.trim().trim_end_matches('/').to_string();
    if !url.contains("/api/embeddings") {
        if url.contains("/api") {
            url.push_str("/embeddings");
        } else {
            if let Some(index) = url.find("/v1") {
                url.truncate(index);
            }
            url.push_str("/api/embeddings");
        }
    }
    url
}

fn handle_openai_embedding_response(
    response: reqwest::blocking::Response,
) -> Result<Vec<Vec<f32>>, AdapterError> {
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(AdapterError::HttpStatus { status, body });
    }

    let parsed: OpenAiEmbeddingResponse = response.json()?;
    let vectors = parsed
        .data
        .into_iter()
        .map(|item| item.embedding.into_iter().map(|v| v as f32).collect())
        .collect::<Vec<Vec<f32>>>();
    if vectors.is_empty() {
        return Err(AdapterError::EmptyResponse);
    }
    Ok(vectors)
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct GeminiEmbeddingResponse {
    embedding: Option<Vec<f64>>,
}
