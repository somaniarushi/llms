import os
import time
from typing import Any, Optional

import requests

from models.base import SamplerBase
from typings import MessageList

MAX_ALLOWED_TRIALS = 7

URL = "https://api.together.xyz/v1/chat/completions"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {os.environ.get('TOGETHER_BEARER_TOKEN')}",
}


class LlamaSampler(SamplerBase):
    """
    Sample from Together's llama3 chat completion API
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        while trial < MAX_ALLOWED_TRIALS:
            try:
                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stop": ["|<im_end>|"],
                }
                response = requests.post(URL, json=payload, headers=HEADERS)
                assert (
                    response.status_code == 200
                ), f"Rate limit detected: {response.text}"
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1


class Llama3_70BSampler(LlamaSampler):
    """
    Sample from Together's llama3 chat completion API with the 370B model
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        super().__init__(
            model="meta-llama/Llama-3-70b-chat-hf",
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class Llama3_70BPreTrainSampler(LlamaSampler):
    """
    Sample from Together's llama3 chat completion API with the 370B model
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        super().__init__(
            model="meta-llama/Llama-3-70b-hf",
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class Llama3_8BSampler(LlamaSampler):
    """
    Sample from Together's llama3 chat completion API with the 8B model
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        super().__init__(
            model="meta-llama/Llama-3-8b-chat-hf",
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class Llama3_8BPreTrainSampler(LlamaSampler):
    """
    Sample from Together's llama3 chat completion API with the 8B model
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        super().__init__(
            model="meta-llama/Llama-3-8b-hf",
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
