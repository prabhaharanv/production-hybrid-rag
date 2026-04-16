from openai import OpenAI


class QueryRewriter:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def rewrite(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query rewriter for a retrieval system. "
                        "Given a user question, rewrite it to be more specific and search-friendly. "
                        "Output ONLY the rewritten query, nothing else. "
                        "Do not answer the question. Do not add explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
