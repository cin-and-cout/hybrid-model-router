import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# We use the fireworks package if installed, but handle import errors or mock usage gracefully
try:
    from fireworks.client import Fireworks
except ImportError:
    Fireworks = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class RemoteExecutionResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Dict[str, Any] = Field(default_factory=dict)

class RemoteClient:
    """
    Client for querying remote chat completion APIs (OpenAI, Gemini, Fireworks),
    with explicit tracking of token consumption.
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
        provider: Optional[str] = None
    ):
        self.model = model
        
        # Auto-detect provider
        if provider:
            self.provider = provider.lower()
        elif "gpt" in model or "text-davinci" in model or "o1-" in model:
            self.provider = "openai"
        elif "gemini" in model:
            self.provider = "gemini"
        elif "groq" in model:
            self.provider = "groq"
        else:
            self.provider = "fireworks"
            
        # Resolve the API key based on the provider
        if self.provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif self.provider == "groq":
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        else:
            self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
            
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                if self.provider == "fireworks":
                    raise ValueError(
                        "FIREWORKS_API_KEY environment variable is not set and no API key was provided."
                    )
                else:
                    raise ValueError(
                        f"API key for provider '{self.provider}' is not set. Please provide it or set the environment variable."
                    )
            
            if OpenAI is None:
                raise ImportError(
                    "The 'openai' package is not installed or could not be imported."
                )
            if self.provider == "gemini":
                # Gemini OpenAI compatibility endpoint
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/"
                )
            elif self.provider == "groq":
                # Groq OpenAI compatibility endpoint
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            elif self.provider == "fireworks":
                if Fireworks is not None:
                    self._client = Fireworks(api_key=self.api_key)
                else:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.fireworks.ai/inference/v1"
                    )
            else:
                self._client = OpenAI(api_key=self.api_key)
        return self._client

    def query(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> RemoteExecutionResult:
        """
        Queries the Fireworks AI remote chat completion endpoint.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        # Call the Fireworks SDK chat completions endpoint with automatic retries
        import time
        max_retries = 3
        backoff_factor = 2.0
        
        # Add request timeout to payload if not already set (typically 'timeout' is accepted by the SDK)
        payload["timeout"] = 30.0
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**payload)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(backoff_factor ** attempt)
        
        text = ""
        if completion.choices:
            text = completion.choices[0].message.content or ""
            
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        usage = getattr(completion, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            
        # Convert completion object to a dict representation for raw response tracking
        # The completion model usually has a dict() or model_dump() method, or we can use custom dict
        raw_dict = {}
        try:
            if hasattr(completion, "model_dump"):
                raw_dict = completion.model_dump()
            elif hasattr(completion, "dict"):
                raw_dict = completion.dict()
            else:
                raw_dict = dict(completion)
        except Exception:
            raw_dict = {"id": getattr(completion, "id", None), "model": getattr(completion, "model", None)}
            
        return RemoteExecutionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=raw_dict
        )
