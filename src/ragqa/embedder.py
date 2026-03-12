import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # OFFLINE指定時はローカルキャッシュのみ参照する
        offline = os.environ.get("TRANSFORMERS_OFFLINE", "0").strip() == "1"
        self.model = SentenceTransformer(model_name, local_files_only=offline)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # float32でFAISS向け
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")
