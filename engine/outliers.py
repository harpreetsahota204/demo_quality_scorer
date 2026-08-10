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
    # "Anomalous" is only defined against a population. One episode has no
    # population, so NaN (unavailable) rather than 0 (normal).
    n = len(feature_matrix)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    x = _impute_columns(np.asarray(feature_matrix, dtype=np.float64))

    # Negated so both scores share the codebase-wide "higher == worse"
    # convention that normalize.py and the panel assume.
    forest = IsolationForest(contamination="auto", random_state=random_state)
    forest.fit(x)
    iforest_scores = -forest.score_samples(x)  # score_samples: higher == more normal

    # Small corpora can have fewer episodes than the requested neighbor count;
    # clamping keeps the metric defined (as a coarser estimate) instead of
    # raising and losing the family entirely.
    k = min(n_neighbors, n - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(x)
    dists, _ = neighbors.kneighbors(x)
    knn_dists = dists[:, 1:].mean(axis=1)  # column 0 is each point's distance to itself

    return iforest_scores, knn_dists


def _impute_columns(x):
    """Fills NaNs with their column median so a missing metric reads as typical.

    Neither model accepts NaN, and dropping the row would silently exclude
    episodes from scoring. Imputing the median is the neutral choice: a metric
    that couldn't be computed on an episode then contributes no evidence
    either way, rather than pushing the episode toward or away from anomalous.
    An all-NaN column falls back to 0, making it inert once every row is equal.
    """
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    nan_rows, nan_cols = np.where(np.isnan(x))
    x = x.copy()
    x[nan_rows, nan_cols] = medians[nan_cols]
    return x
