from openai import OpenAI


class LLMGenerator:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise question-answering assistant. "
                        "You answer ONLY using the retrieved context provided. "
                        "You NEVER fabricate information. "
                        "If the context is insufficient, respond with exactly INSUFFICIENT_CONTEXT."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()