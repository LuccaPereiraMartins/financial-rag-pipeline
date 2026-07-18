"""OpenAI embedding client wrapper."""

from openai import OpenAI

from src.config import Config


class Embedder:
    def __init__(self):
        # OpenAI() reads OPENAI_API_KEY from the environment by default
        # you could use a global OpenAI client or async one if needed
        self.client = OpenAI()
        self.model = Config.EMBEDDING_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # NOTE: batching supported, cost efficient and faster
        # and should prevent rate limiting
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            # NOTE: Ensure order by index, they could come back out of order if batched
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([d.embedding for d in ordered])
        return out

    def embed_one(self, text: str) -> list[float]:
        # needed for single text inputs, such as the tool calls
        return self.embed([text])[0]
