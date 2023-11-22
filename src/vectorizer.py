import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", cache_folder="/model_cache"
        )

    def vectorize(self, text: list[str]) -> list[np.ndarray]:
        if len(text) == 0:
            return []
        return self.model.encode(
            sentences=text,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
