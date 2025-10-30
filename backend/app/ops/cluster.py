from __future__ import annotations

import logging
from typing import Optional

import hdbscan
import numpy as np
import umap


logger = logging.getLogger(__name__)


def umap_reduce(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    n_samples = embeddings.shape[0]
    if embeddings.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_samples <= 1:
        return np.zeros((n_samples, 2), dtype=np.float32)

    effective_neighbors = max(int(n_neighbors), 2)
    if n_samples <= effective_neighbors:
        effective_neighbors = max(n_samples - 1, 1)

    reducer = umap.UMAP(
        n_neighbors=effective_neighbors,
        n_components=2,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    try:
        coords = reducer.fit_transform(embeddings)
    except Exception:  # pragma: no cover - defensive fallback for tiny datasets
        logger.warning("UMAP failed; returning default layout", exc_info=True)
        coords = _fallback_layout(n_samples)
    return np.asarray(coords, dtype=np.float32)


def hdbscan_cluster(
    points: np.ndarray,
    *,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    n_samples = points.shape[0]
    if points.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if n_samples <= 1:
        return np.full((n_samples,), -1, dtype=np.int32)

    kwargs = {
        "min_cluster_size": max(2, min(int(min_cluster_size), n_samples)),
        "metric": metric,
    }
    if min_samples is not None:
        kwargs["min_samples"] = max(1, min(int(min_samples), n_samples))

    clusterer = hdbscan.HDBSCAN(**kwargs)
    try:
        labels = clusterer.fit_predict(points)
    except Exception:  # pragma: no cover - defensive fallback for tiny datasets
        logger.warning("HDBSCAN failed; marking all points as noise", exc_info=True)
        return np.full((n_samples,), -1, dtype=np.int32)
    return np.asarray(labels, dtype=np.int32)


def _fallback_layout(n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32)
    angles = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
