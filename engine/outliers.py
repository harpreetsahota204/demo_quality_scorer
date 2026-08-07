"""Corpus-wide outlier detection: isolation forest + kNN manifold distance.

Both need more than one episode to fit against, so these operate on a whole
batch's feature matrix at once (see :mod:`.score`), not on a single episode
in isolation.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

N_NEIGHBORS = 5


def fit_and_score(feature_matrix, n_neighbors=N_NEIGHBORS, random_state=0):
    """Fits an isolation forest and a kNN model on a corpus of feature vectors.

    Args:
        feature_matrix: an ``(n_episodes, n_features)`` array; NaNs are
            replaced with each feature's column median before fitting
        n_neighbors (5): neighbors to average for the manifold-distance score
        random_state (0): seed for the isolation forest

    Returns:
        a tuple ``(iforest_scores, knn_dists)``, each an ``(n_episodes,)``
        array where higher means more anomalous
    """
    n = len(feature_matrix)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    x = _impute_columns(np.asarray(feature_matrix, dtype=np.float64))

    forest = IsolationForest(contamination="auto", random_state=random_state)
    forest.fit(x)
    iforest_scores = -forest.score_samples(x)  # score_samples: higher == more normal

    k = min(n_neighbors, n - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(x)
    dists, _ = neighbors.kneighbors(x)
    knn_dists = dists[:, 1:].mean(axis=1)  # column 0 is each point's distance to itself

    return iforest_scores, knn_dists


def _impute_columns(x):
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    nan_rows, nan_cols = np.where(np.isnan(x))
    x = x.copy()
    x[nan_rows, nan_cols] = medians[nan_cols]
    return x
