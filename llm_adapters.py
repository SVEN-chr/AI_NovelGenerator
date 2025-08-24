# llm_adapters.py
# -*- coding: utf-8 -*-
import logging
from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
# from google import genai
import google.generativeai as genai
# from google.genai import types
from google.generativeai import types
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from openai import OpenAI
import requests


def check_base_url(url: str) -> str:
    """
    处理base_url的规则：
    1. 如果url以#结尾，则移除#并直接使用用户提供的url
    2. 否则检查是否需要添加/v1后缀
    """
    import re
    url = url.strip()
    if not url:
        return url
        
    if url.endswith('#'):
        return url.rstrip('#')
        
    if not re.search(r'/v\d+$', url):
        if '/v1' not in url:
            url = url.rstrip('/') + '/v1'
    return url

class BaseLLMAdapter:
    """
    统一的 LLM 接口基类，为不同后端（OpenAI、Ollama、ML Studio、Gemini等）提供一致的方法签名。
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement .invoke(prompt) method.")

class DeepSeekAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from DeepSeekAdapter.")
            return ""
        return response.content

class OpenAIAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OpenAIAdapter.")
            return ""
        return response.content

class GeminiAdapter(BaseLLMAdapter):
    """
    适配 Google Gemini (Google Generative AI) 接口
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        # gemini超时时间是毫秒
        self.timeout = timeout * 1000

        # Configure the Gemini API
        # Note: base_url is typically not used with standard Gemini API
        # but we store it for potential future use or custom endpoints
        self.base_url = base_url
        genai.configure(api_key=self.api_key)
        
        # Configure safety settings to be less restrictive for creative writing
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self._client = genai.GenerativeModel(
            self.model_name,
            safety_settings=safety_settings
        )

    def invoke(self, prompt: str) -> str:
        import time
        import re
        
        max_retries = 3
        base_delay = 5  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Create generation config
                generation_config = genai.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                
                response = self._client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    # Extract retry delay from error message if available
                    retry_match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
                    if retry_match:
                        delay = int(retry_match.group(1)) + 5  # Add 5 seconds buffer
                    else:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                    
                    if attempt < max_retries - 1:
                        logging.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logging.error(f"Rate limit exceeded after {max_retries} attempts: {e}")
                        return ""
                else:
                    # Non-rate-limit error, don't retry
                    logging.error(f"Gemini API 调用失败: {e}")
                    return ""
        
        # Better response validation
        if response:
            # Check if response was blocked by safety filters
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 1:  # STOP
                        pass  # Normal completion
                    elif finish_reason == 2:  # MAX_TOKENS
                        logging.warning("Gemini response truncated due to max_tokens limit")
                    elif finish_reason == 3:  # SAFETY
                        logging.warning("Gemini response blocked by safety filters")
                        return ""
                    elif finish_reason == 4:  # RECITATION
                        logging.warning("Gemini response blocked due to recitation concerns")
                        return ""
                    else:
                        logging.warning(f"Gemini response finished with reason: {finish_reason}")
            
            # Try to get text content safely with detailed debugging
            try:
                # First try the direct text accessor
                if hasattr(response, 'text'):
                    text_content = response.text
                    if text_content:
                        return text_content
                
                # Debug response structure
                logging.debug(f"Response object type: {type(response)}")
                logging.debug(f"Response attributes: {dir(response)}")
                
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    logging.debug(f"Candidate type: {type(candidate)}")
                    logging.debug(f"Candidate attributes: {dir(candidate)}")
                    
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        logging.debug(f"Content type: {type(content)}")
                        logging.debug(f"Content attributes: {dir(content)}")
                        
                        if hasattr(content, 'parts') and content.parts:
                            logging.debug(f"Number of parts: {len(content.parts)}")
                            text_parts = []
                            for i, part in enumerate(content.parts):
                                logging.debug(f"Part {i} type: {type(part)}")
                                logging.debug(f"Part {i} attributes: {dir(part)}")
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                                    logging.debug(f"Part {i} text length: {len(part.text)}")
                            
                            if text_parts:
                                result = ''.join(text_parts)
                                logging.debug(f"Extracted text length: {len(result)}")
                                return result
                            else:
                                logging.warning("No text found in any parts")
                        else:
                            logging.warning("No parts found in content")
                    else:
                        logging.warning("No content found in candidate")
                else:
                    logging.warning("No candidates found in response")
                        
            except Exception as text_ex:
                logging.error(f"Error extracting text from Gemini response: {text_ex}")
                # Try alternative approach - check if it's a generator response
                try:
                    if hasattr(response, '_result') and hasattr(response._result, 'candidates'):
                        candidates = response._result.candidates
                        if candidates and len(candidates) > 0:
                            candidate = candidates[0]
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    text_parts = []
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            text_parts.append(part.text)
                                    if text_parts:
                                        return ''.join(text_parts)
                except Exception as alt_ex:
                    logging.error(f"Alternative text extraction also failed: {alt_ex}")
            
            logging.warning("No valid text content found in Gemini API response")
            return ""
        else:
            logging.warning("No response from Gemini API")
            return ""

class AzureOpenAIAdapter(BaseLLMAdapter):
    """
    适配 Azure OpenAI 接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        import re
        match = re.match(r'https://(.+?)/openai/deployments/(.+?)/chat/completions\?api-version=(.+)', base_url)
        if match:
            self.azure_endpoint = f"https://{match.group(1)}"
            self.azure_deployment = match.group(2)
            self.api_version = match.group(3)
        else:
            raise ValueError("Invalid Azure OpenAI base_url format")
        
        self.api_key = api_key
        self.model_name = self.azure_deployment
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from AzureOpenAIAdapter.")
            return ""
        return response.content

class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama 同样有一个 OpenAI-like /v1/chat 接口，可直接使用 ChatOpenAI。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        if self.api_key == '':
            self.api_key= 'ollama'

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OllamaAdapter.")
            return ""
        return response.content

class MLStudioAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.invoke(prompt)
            if not response:
                logging.warning("No response from MLStudioAdapter.")
                return ""
            return response.content
        except Exception as e:
            logging.error(f"ML Studio API 调用超时或失败: {e}")
            return ""

class AzureAIAdapter(BaseLLMAdapter):
    """
    适配 Azure AI Inference 接口，用于访问Azure AI服务部署的模型
    使用 azure-ai-inference 库进行API调用
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        import re
        # 匹配形如 https://xxx.services.ai.azure.com/models/chat/completions?api-version=xxx 的URL
        match = re.match(r'https://(.+?)\.services\.ai\.azure\.com(?:/models)?(?:/chat/completions)?(?:\?api-version=(.+))?', base_url)
        if match:
            # endpoint需要是形如 https://xxx.services.ai.azure.com/models 的格式
            self.endpoint = f"https://{match.group(1)}.services.ai.azure.com/models"
            # 如果URL中包含api-version参数，使用它；否则使用默认值
            self.api_version = match.group(2) if match.group(2) else "2024-05-01-preview"
        else:
            raise ValueError("Invalid Azure AI base_url format. Expected format: https://<endpoint>.services.ai.azure.com/models/chat/completions?api-version=xxx")
        
        self.base_url = self.endpoint  # 存储处理后的endpoint URL
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.complete(
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage(prompt)
                ]
            )
            if response and response.choices:
                return response.choices[0].message.content
            else:
                logging.warning("No response from AzureAIAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Azure AI Inference API 调用失败: {e}")
            return ""

# 火山引擎实现
class VolcanoEngineAIAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout  # 添加超时配置
        )
    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.timeout  # 添加超时参数
            )
            if not response:
                logging.warning("No response from DeepSeekAdapter.")
                return ""
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"火山引擎API调用超时或失败: {e}")
            return ""

class SiliconFlowAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout  # 添加超时配置
        )
    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.timeout  # 添加超时参数
            )
            if not response:
                logging.warning("No response from DeepSeekAdapter.")
                return ""
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"硅基流动API调用超时或失败: {e}")
            return ""
# grok實現
class GrokAdapter(BaseLLMAdapter):
    """
    适配 xAI Grok API
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are Grok, created by xAI."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if response and response.choices:
                return response.choices[0].message.content
            else:
                logging.warning("No response from GrokAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Grok API 调用失败: {e}")
            return ""

def create_llm_adapter(
    interface_format: str,
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int
) -> BaseLLMAdapter:
    """
    工厂函数：根据 interface_format 返回不同的适配器实例。
    """
    fmt = interface_format.strip().lower()
    if fmt == "deepseek":
        return DeepSeekAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "openai":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "azure openai":
        return AzureOpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "azure ai":
        return AzureAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "ollama":
        return OllamaAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "ml studio":
        return MLStudioAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "gemini":
        return GeminiAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "阿里云百炼":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "火山引擎":
        return VolcanoEngineAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "硅基流动":
        return SiliconFlowAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "grok":
        return GrokAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    else:
        raise ValueError(f"Unknown interface_format: {interface_format}")
