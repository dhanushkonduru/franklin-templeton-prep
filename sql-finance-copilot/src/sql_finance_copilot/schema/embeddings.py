from __future__ import annotations

from functools import lru_cache

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.astype("float32")


@lru_cache(maxsize=1)
def embed_texts(model_name: str, texts_tuple: tuple[str, ...]) -> np.ndarray:
    return SentenceTransformerEmbedder(model_name).embed(list(texts_tuple))
