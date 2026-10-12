"""Thought embedding for semantic loop detection.

Default: TF-IDF + truncated SVD. Deterministic, zero downloads, CPU-only.
A sentence-transformer backend was evaluated and rejected: it requires a
~90MB model download and gives no measured advantage at the trajectory
scale (tens of thoughts) where loopguard operates. Any callable with
signature embed(list[str]) -> np.ndarray can be injected instead.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


class ThoughtEmbedder:
    """Embed agent thoughts into a fixed-size semantic space.

    Word-level TF-IDF (captures topic) + character n-gram TF-IDF (captures
    morphological/paraphrase similarity), fused and compressed with SVD.
    """

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                     sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                                     sublinear_tf=True, max_features=4000)),
        ])
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, thoughts: list[str]) -> "ThoughtEmbedder":
        corpus = [t if t.strip() else "<empty>" for t in thoughts]
        X = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.n_components, max(1, X.shape[1] - 1), max(1, X.shape[0] - 1))
        if n_comp < self.n_components:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(X)
        return self

    def embed(self, thoughts: list[str] | str) -> np.ndarray:
        single = isinstance(thoughts, str)
        items = [thoughts] if single else list(thoughts)
        items = [t if t.strip() else "<empty>" for t in items]
        X = self.vectorizer.transform(items)
        E = self.svd.transform(X)
        E = normalize(E, norm="l2")
        # Rows that were all-OOV come back as zero vectors; leave them zero.
        return E[0] if single else E

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
