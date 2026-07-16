"""OpenAI embedding client wrapper."""

from openai import OpenAI

from src.config import Config


class Embedder:
    def __init__(self):
        # OpenAI() reads OPENAI_API_KEY from the environment by default
        self.client = OpenAI()
        self.model = Config.EMBEDDING_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = [t if t.strip() else " " for t in texts[i : i + 64]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([d.embedding for d in ordered])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
