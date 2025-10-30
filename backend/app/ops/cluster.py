from __future__ import annotations

from typing import Optional

import hdbscan
import numpy as np
import umap


def umap_reduce(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if embeddings.shape[0] <= 1:
        return np.zeros((embeddings.shape[0], 2), dtype=np.float32)

    reducer = umap.UMAP(
        n_neighbors=max(int(n_neighbors), 2),
        n_components=2,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    return np.asarray(coords, dtype=np.float32)


def hdbscan_cluster(
    points: np.ndarray,
    *,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.int32)

    kwargs = {
        "min_cluster_size": max(int(min_cluster_size), 2),
        "metric": metric,
    }
    if min_samples is not None:
        kwargs["min_samples"] = max(int(min_samples), 1)

    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels = clusterer.fit_predict(points)
    return np.asarray(labels, dtype=np.int32)
