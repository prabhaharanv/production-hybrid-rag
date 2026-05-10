from collections.abc import Generator as GenIterator

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "You answer ONLY using the retrieved context provided. "
    "You NEVER fabricate information. "
    "If the context is insufficient, respond with exactly INSUFFICIENT_CONTEXT."
)


class LLMGenerator:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _messages(self, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt: str) -> GenIterator[str, None, None]:
        """Yield token chunks from the LLM as they arrive."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
            temperature=0.1,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
