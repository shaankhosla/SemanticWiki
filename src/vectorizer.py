import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.decomposition import PCA  # type: ignore


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

    def dimensionality_reduction(
        self, vectors: list[np.ndarray], components: int = 2
    ) -> np.ndarray:
        if len(vectors) < components:
            raise Exception(
                f"Number of vectors {len(vectors)} not greater than components {components}"
            )
        pca = PCA(n_components=components)
        return pca.fit_transform(vectors)
