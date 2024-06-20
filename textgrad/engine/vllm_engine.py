import requests
import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import json
from .base import EngineLM, CachedEngine

class LLMEngine(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    PROMPT_TEMPLATE = "<|user|>\n{user}\n<|end|>\n<|assistant|>\n"

    def __init__(
        self,
        model_url="localhost:8000/",
        model_name="test_model"
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_name}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.model_name = model_name
        self.model_url=model_url



    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = requests.post(
            url=self.model_url,
            json={
                "prompt":prompt,
                "temperature":temperature,
                "top_p":top_p,
                "max_tokens":max_tokens
            }
        )

        response = response.json["output"]
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
