import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore
<<<<<<< HEAD
from sklearn.decomposition import PCA  # type: ignore
=======
>>>>>>> 8ffd7f5 (added PCA and t-SNA)


class Vectorizer:
    def __init__(self):
<<<<<<< HEAD
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", cache_folder="/model_cache"
        )
=======
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
>>>>>>> 8ffd7f5 (added PCA and t-SNA)

    def vectorize(self, text: list[str]) -> list[np.ndarray]:
        if len(text) == 0:
            return []
        return self.model.encode(
            sentences=text,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
<<<<<<< HEAD

    def dimensionality_reduction(
        self, vectors: list[np.ndarray], components: int = 2
    ) -> np.ndarray:
        pca = PCA(n_components=components)
        return pca.fit_transform(vectors)
=======
>>>>>>> 8ffd7f5 (added PCA and t-SNA)
