import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# We use the fireworks package if installed, but handle import errors or mock usage gracefully
try:
    from fireworks.client import Fireworks
except ImportError:
    Fireworks = None

class RemoteExecutionResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Dict[str, Any] = Field(default_factory=dict)

class RemoteClient:
    """
    Client for querying the remote Fireworks AI chat completions API,
    with explicit tracking of token consumption.
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    ):
        # Resolve the API key from argument or environment variable
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.model = model
        
        if not self.api_key:
            # We don't raise immediately during initialization so we can instantiate the class
            # in tests or when mocking, but we will check it before making a live request.
            pass
            
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "FIREWORKS_API_KEY environment variable is not set and no API key was provided."
                )
            if Fireworks is None:
                raise ImportError(
                    "The 'fireworks-ai' package is not installed or could not be imported."
                )
            self._client = Fireworks(api_key=self.api_key)
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
            
        # Call the Fireworks SDK chat completions endpoint
        completion = self.client.chat.completions.create(**payload)
        
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
