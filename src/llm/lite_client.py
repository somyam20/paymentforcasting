import os
from litellm import completion
from config.settings import LITE_LLM_API_KEY, LITE_LLM_MODEL

class LiteClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or LITE_LLM_API_KEY
        self.model = model or LITE_LLM_MODEL

        if not self.api_key:
            raise ValueError("LITE_LLM_API_KEY not set")

    def generate(self, prompt: str, max_tokens: int = 512):
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]

lite_client = LiteClient()
