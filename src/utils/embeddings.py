from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Centralized embedding service.
    Agents never create models directly.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into normalized embeddings.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )
