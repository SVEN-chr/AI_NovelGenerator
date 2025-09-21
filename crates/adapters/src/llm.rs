use std::thread;
use std::time::Duration;

use log::warn;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::blocking::Client;
use reqwest::header::{self, HeaderMap, HeaderValue};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use novel_core::config::{Config, LlmConfig};

use crate::base_url::check_base_url;
use crate::error::AdapterError;
use crate::retry::{call_with_retry, RetryConfig};

pub trait LanguageModel: Send + Sync {
    fn invoke(&self, prompt: &str) -> Result<String, AdapterError>;
}

pub fn create_llm_adapter(
    config: &Config,
    profile_name: &str,
) -> Result<Box<dyn LanguageModel>, AdapterError> {
    let profile = config.get_llm_profile(profile_name).ok_or_else(|| {
        AdapterError::InvalidConfig(format!("unknown LLM profile `{}`", profile_name))
    })?;
    create_llm_adapter_from_profile(profile)
}

pub fn create_llm_adapter_from_profile(
    profile: &LlmConfig,
) -> Result<Box<dyn LanguageModel>, AdapterError> {
    let fmt = profile.interface_format.trim().to_lowercase();
    let timeout = profile.timeout.max(1);

    match fmt.as_str() {
        "openai" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, "https://api.openai.com/v1"),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("You are a helpful assistant.".to_string()),
        )?)),
        "deepseek" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, "https://api.deepseek.com/v1"),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("You are a helpful assistant.".to_string()),
        )?)),
        "ollama" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, "http://localhost:11434/v1"),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("You are a helpful assistant.".to_string()),
        )?)),
        "ml studio" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, "http://localhost:5000/v1"),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("You are a helpful assistant.".to_string()),
        )?)),
        "阿里云百炼" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, ""),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("You are a helpful assistant.".to_string()),
        )?)),
        "火山引擎" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, ""),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("你是DeepSeek，是一个 AI 人工智能助手".to_string()),
        )?)),
        "硅基流动" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, ""),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("你是DeepSeek，是一个 AI 人智能助手".to_string()),
        )?)),
        "grok" => Ok(Box::new(OpenAiLikeAdapter::new(
            resolve_base_url(&profile.base_url, "https://api.x.ai/v1"),
            optional_string(&profile.api_key),
            profile.model_name.clone(),
            profile.max_tokens,
            profile.temperature,
            timeout,
            Some("You are Grok, created by xAI.".to_string()),
        )?)),
        "azure openai" => Ok(Box::new(AzureOpenAiAdapter::new(
            profile.api_key.clone(),
            &profile.base_url,
            profile.max_tokens,
            profile.temperature,
            timeout,
        )?)),
        "azure ai" => Ok(Box::new(AzureAiAdapter::new(
            profile.api_key.clone(),
            &profile.base_url,
            &profile.model_name,
            profile.max_tokens,
            profile.temperature,
            timeout,
        )?)),
        "gemini" => Ok(Box::new(GeminiAdapter::new(
            profile.api_key.clone(),
            &profile.base_url,
            &profile.model_name,
            profile.max_tokens,
            profile.temperature,
            timeout,
        )?)),
        other => Err(AdapterError::InvalidConfig(format!(
            "unknown interface_format: {}",
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

fn resolve_base_url(base_url: &str, default: &str) -> String {
    let raw = if base_url.trim().is_empty() {
        default.to_string()
    } else {
        base_url.to_string()
    };
    check_base_url(&raw)
}

struct OpenAiLikeAdapter {
    client: Client,
    url: String,
    api_key: Option<String>,
    model_name: String,
    max_tokens: Option<u32>,
    temperature: f32,
    system_prompt: Option<String>,
    retry: RetryConfig,
}

impl OpenAiLikeAdapter {
    fn new(
        base_url: String,
        api_key: Option<String>,
        model_name: String,
        max_tokens: u32,
        temperature: f32,
        timeout: u64,
        system_prompt: Option<String>,
    ) -> Result<Self, AdapterError> {
        if base_url.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "base_url must not be empty".to_string(),
            ));
        }

        if model_name.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "model_name must not be empty".to_string(),
            ));
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()?;

        Ok(Self {
            client,
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model_name,
            max_tokens: if max_tokens == 0 {
                None
            } else {
                Some(max_tokens)
            },
            temperature,
            system_prompt,
            retry: RetryConfig::default(),
        })
    }

    fn invoke_once(&self, prompt: &str) -> Result<String, AdapterError> {
        let mut messages: Vec<ChatMessageRequest<'_>> = Vec::new();
        if let Some(system) = self.system_prompt.as_deref() {
            messages.push(ChatMessageRequest {
                role: "system",
                content: system,
            });
        }
        messages.push(ChatMessageRequest {
            role: "user",
            content: prompt,
        });

        let body = ChatCompletionRequest {
            model: Some(self.model_name.as_str()),
            messages,
            max_tokens: self.max_tokens,
            max_output_tokens: None,
            temperature: Some(self.temperature),
        };

        let mut request = self.client.post(&self.url).header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                request = request.bearer_auth(key);
            }
        }

        let response = request.json(&body).send()?;
        handle_chat_response(response)
    }
}

impl LanguageModel for OpenAiLikeAdapter {
    fn invoke(&self, prompt: &str) -> Result<String, AdapterError> {
        call_with_retry(|| self.invoke_once(prompt), &self.retry)
    }
}

struct AzureOpenAiAdapter {
    client: Client,
    url: String,
    api_key: String,
    max_tokens: Option<u32>,
    temperature: f32,
    retry: RetryConfig,
}

impl AzureOpenAiAdapter {
    fn new(
        api_key: String,
        base_url: &str,
        max_tokens: u32,
        temperature: f32,
        timeout: u64,
    ) -> Result<Self, AdapterError> {
        if api_key.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Azure OpenAI api_key must not be empty".to_string(),
            ));
        }

        static AZURE_RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(
                r"^https://([^/]+)/openai/deployments/([^/]+)/chat/completions\?api-version=([^/?&]+)"
            )
            .unwrap()
        });

        let base = base_url.trim();
        let captures = AZURE_RE.captures(base).ok_or_else(|| {
            AdapterError::InvalidConfig(
                "Invalid Azure OpenAI base_url format. Expected https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>"
                    .to_string(),
            )
        })?;

        let endpoint = format!("https://{}", &captures[1]);
        let deployment = captures[2].to_string();
        let api_version = captures[3].to_string();

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()?;

        Ok(Self {
            client,
            url: format!(
                "{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            ),
            api_key,
            max_tokens: if max_tokens == 0 { None } else { Some(max_tokens) },
            temperature,
            retry: RetryConfig::default(),
        })
    }

    fn invoke_once(&self, prompt: &str) -> Result<String, AdapterError> {
        let messages = vec![ChatMessageRequest {
            role: "user",
            content: prompt,
        }];

        let body = ChatCompletionRequest {
            model: None,
            messages,
            max_tokens: self.max_tokens,
            max_output_tokens: None,
            temperature: Some(self.temperature),
        };

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

        let response = self
            .client
            .post(&self.url)
            .headers(headers)
            .json(&body)
            .send()?;
        handle_chat_response(response)
    }
}

impl LanguageModel for AzureOpenAiAdapter {
    fn invoke(&self, prompt: &str) -> Result<String, AdapterError> {
        call_with_retry(|| self.invoke_once(prompt), &self.retry)
    }
}

struct AzureAiAdapter {
    client: Client,
    url: String,
    api_key: String,
    max_tokens: Option<u32>,
    temperature: f32,
    retry: RetryConfig,
}

impl AzureAiAdapter {
    fn new(
        api_key: String,
        base_url: &str,
        model_name: &str,
        max_tokens: u32,
        temperature: f32,
        timeout: u64,
    ) -> Result<Self, AdapterError> {
        if api_key.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Azure AI api_key must not be empty".to_string(),
            ));
        }

        static AZURE_AI_RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(
                r"^https://([^.]+)\.services\.ai\.azure\.com(?:/models)?(?:/chat/completions)?(?:\?api-version=([^&]+))?"
            )
            .unwrap()
        });

        let base = base_url.trim();
        let captures = AZURE_AI_RE.captures(base).ok_or_else(|| {
            AdapterError::InvalidConfig(
                "Invalid Azure AI base_url format. Expected https://<endpoint>.services.ai.azure.com/models/<model>/chat/completions?api-version=xxx"
                    .to_string(),
            )
        })?;

        let endpoint = format!("https://{}.services.ai.azure.com", &captures[1]);
        let api_version = captures
            .get(2)
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "2024-05-01-preview".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()?;

        Ok(Self {
            client,
            url: format!(
                "{endpoint}/models/{model_name}/chat/completions?api-version={api_version}"
            ),
            api_key,
            max_tokens: if max_tokens == 0 {
                None
            } else {
                Some(max_tokens)
            },
            temperature,
            retry: RetryConfig::default(),
        })
    }

    fn invoke_once(&self, prompt: &str) -> Result<String, AdapterError> {
        let messages = vec![ChatMessageRequest {
            role: "user",
            content: prompt,
        }];

        let body = ChatCompletionRequest {
            model: None,
            messages,
            max_tokens: self.max_tokens,
            max_output_tokens: self.max_tokens,
            temperature: Some(self.temperature),
        };

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

        let response = self
            .client
            .post(&self.url)
            .headers(headers)
            .json(&body)
            .send()?;
        handle_chat_response(response)
    }
}

impl LanguageModel for AzureAiAdapter {
    fn invoke(&self, prompt: &str) -> Result<String, AdapterError> {
        call_with_retry(|| self.invoke_once(prompt), &self.retry)
    }
}

struct GeminiAdapter {
    client: Client,
    url: String,
    temperature: f32,
    max_tokens: u32,
    retry: RetryConfig,
    base_delay: Duration,
}

impl GeminiAdapter {
    fn new(
        api_key: String,
        base_url: &str,
        model_name: &str,
        max_tokens: u32,
        temperature: f32,
        timeout: u64,
    ) -> Result<Self, AdapterError> {
        if api_key.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Gemini api_key must not be empty".to_string(),
            ));
        }

        if model_name.trim().is_empty() {
            return Err(AdapterError::InvalidConfig(
                "Gemini model_name must not be empty".to_string(),
            ));
        }

        let base = if base_url.trim().is_empty() {
            "https://generativelanguage.googleapis.com/v1beta".to_string()
        } else {
            base_url.trim().trim_end_matches('/').to_string()
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()?;

        Ok(Self {
            client,
            url: format!(
                "{base}/models/{model}:generateContent?key={api}",
                model = model_name,
                api = api_key
            ),
            temperature,
            max_tokens,
            retry: RetryConfig::default(),
            base_delay: Duration::from_secs(5),
        })
    }

    fn invoke_once(&self, prompt: &str) -> Result<String, AdapterError> {
        let request = GeminiRequest {
            contents: vec![GeminiRequestContent {
                role: "user",
                parts: vec![GeminiRequestPart { text: prompt }],
            }],
            generation_config: GeminiGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
            },
        };

        let response = self.client.post(&self.url).json(&request).send()?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(AdapterError::HttpStatus { status, body });
        }

        let parsed: GeminiResponse = response.json()?;
        parse_gemini_response(parsed)
    }

    fn rate_limit_delay(&self, err: &AdapterError, attempt: usize) -> Option<Duration> {
        match err {
            AdapterError::HttpStatus { status, body } => {
                let lower = body.to_ascii_lowercase();
                if *status == StatusCode::TOO_MANY_REQUESTS
                    || lower.contains("quota")
                    || lower.contains("rate limit")
                {
                    if let Some(secs) = parse_retry_delay(body) {
                        return Some(Duration::from_secs(secs + 5));
                    }
                    let multiplier = 1u32.checked_shl(attempt as u32).unwrap_or(1);
                    return self
                        .base_delay
                        .checked_mul(multiplier)
                        .or(Some(self.base_delay));
                }
                None
            }
            _ => None,
        }
    }
}

impl LanguageModel for GeminiAdapter {
    fn invoke(&self, prompt: &str) -> Result<String, AdapterError> {
        let mut last_error = None;

        for attempt in 0..self.retry.max_retries {
            match self.invoke_once(prompt) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    let should_retry = attempt + 1 < self.retry.max_retries;
                    if should_retry {
                        if let Some(delay) = self.rate_limit_delay(&err, attempt) {
                            warn!(
                                "Gemini rate limit encountered, retrying in {:?} (attempt {}/{})",
                                delay,
                                attempt + 1,
                                self.retry.max_retries
                            );
                            thread::sleep(delay);
                            last_error = Some(err);
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }

        let err = last_error.unwrap_or(AdapterError::EmptyResponse);
        Err(AdapterError::retry_exhausted(self.retry.max_retries, err))
    }
}

fn handle_chat_response(response: reqwest::blocking::Response) -> Result<String, AdapterError> {
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(AdapterError::HttpStatus { status, body });
    }

    let parsed: ChatCompletionResponse = response.json()?;
    extract_choice_content(parsed).ok_or(AdapterError::EmptyResponse)
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    messages: Vec<ChatMessageRequest<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max_tokens")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max_output_tokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct ChatMessageRequest<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

fn extract_choice_content(response: ChatCompletionResponse) -> Option<String> {
    for choice in response.choices {
        if let Some(message) = choice.message {
            if let Some(content) = message.content {
                if !content.trim().is_empty() {
                    return Some(content);
                }
            }
        }
        if let Some(content) = choice.content {
            if !content.trim().is_empty() {
                return Some(content);
            }
        }
    }
    None
}

#[derive(Serialize)]
struct GeminiRequest<'a> {
    contents: Vec<GeminiRequestContent<'a>>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
}

#[derive(Serialize)]
struct GeminiRequestContent<'a> {
    role: &'static str,
    parts: Vec<GeminiRequestPart<'a>>,
}

#[derive(Serialize)]
struct GeminiRequestPart<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    Other(serde_json::Value),
}

fn parse_gemini_response(response: GeminiResponse) -> Result<String, AdapterError> {
    for candidate in response.candidates {
        if let Some(reason) = candidate.finish_reason.as_deref() {
            match reason {
                "MAX_TOKENS" => warn!("Gemini response truncated due to max_tokens limit"),
                "SAFETY" => warn!("Gemini response blocked by safety filters"),
                "RECITATION" => warn!("Gemini response blocked due to recitation concerns"),
                _ => {}
            }
        }

        if let Some(content) = candidate.content {
            let mut text = String::new();
            for part in content.parts {
                if let GeminiPart::Text { text: part_text } = part {
                    text.push_str(&part_text);
                }
            }
            if !text.trim().is_empty() {
                return Ok(text);
            }
        }
    }

    Err(AdapterError::EmptyResponse)
}

fn parse_retry_delay(body: &str) -> Option<u64> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(details) = value
            .get("error")
            .and_then(|v| v.get("details"))
            .and_then(|v| v.as_array())
        {
            for detail in details {
                if let Some(delay) = detail
                    .get("retryDelay")
                    .or_else(|| detail.get("retry_delay"))
                {
                    if let Some(parsed) = parse_delay_value(delay) {
                        return Some(parsed);
                    }
                }
            }
        }
    }

    static RETRY_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"retry[_ ]?delay[^0-9]*(\d+)").expect("valid regex for retry delay")
    });

    if let Some(caps) = RETRY_RE.captures(body) {
        if let Some(matched) = caps.get(1) {
            if let Ok(value) = matched.as_str().parse::<u64>() {
                return Some(value);
            }
        }
    }

    None
}

fn parse_delay_value(value: &serde_json::Value) -> Option<u64> {
    if let Some(number) = value.as_u64() {
        return Some(number);
    }

    if let Some(text) = value.as_str() {
        if let Ok(number) = text.trim_end_matches('s').parse::<u64>() {
            return Some(number);
        }
    }

    None
}
