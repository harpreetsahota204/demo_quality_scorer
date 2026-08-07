"""Dataset-wide robust normalization (median/MAD z-scores).

Generic over metric names: callers decide which raw per-episode metrics to
normalize, their polarity, and how to combine them. Keeping this module
metric-agnostic means the same normalization logic serves motion, health,
and outlier metrics alike.
"""

import numpy as np

WARN_Z = 2.0
FAIL_Z = 3.0
Z_CLIP = 10.0
_MIN_MAD = 1e-6
_MAD_TO_STD = 1.4826  # scales MAD to be comparable to a normal distribution's std


def fit(values_by_metric):
    """Fits robust ``(median, mad)`` stats for each metric across a corpus.

    Args:
        values_by_metric: a dict of metric name -> array-like of raw values,
            one per episode (NaNs are ignored)

    Returns:
        a dict of metric name -> ``{"median": float, "mad": float}``
    """
    stats = {}
    for metric, values in values_by_metric.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            stats[metric] = {"median": 0.0, "mad": _MIN_MAD}
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        stats[metric] = {"median": median, "mad": max(mad, _MIN_MAD)}
    return stats


def zscore(value, metric_stats, higher_is_worse=True):
    """Robust z-score of one raw value against fitted ``(median, mad)`` stats.

    The sign is flipped when ``higher_is_worse`` is False, so the result is
    always positive-means-worse regardless of the metric's own polarity.
    The result is clipped to +/- :data:`Z_CLIP`: many metrics (e.g. health
    ones) are exactly zero for a clean majority of episodes, which floors
    their MAD near-zero, and without a clip a single small-but-real nonzero
    value would otherwise blow up to an arbitrarily huge z-score and
    dominate any aggregate built from several metrics.
    """
    if value is None or np.isnan(value):
        return np.nan
    z = (value - metric_stats["median"]) / (_MAD_TO_STD * metric_stats["mad"])
    z = z if higher_is_worse else -z
    return float(np.clip(z, -Z_CLIP, Z_CLIP))


def raw_value_at_z(z, metric_stats, higher_is_worse=True):
    """Inverse of :func:`zscore` (ignoring the clip): the raw value whose oriented z-score is ``z``.

    Lets a UI draw thresholds in a metric's own units -- e.g. the raw value
    where warn severity starts is ``raw_value_at_z(WARN_Z, stats, polarity)``.
    """
    delta = z * _MAD_TO_STD * metric_stats["mad"]
    return float(metric_stats["median"] + (delta if higher_is_worse else -delta))


def aggregate(z_by_metric, weights=None):
    """Combines already-oriented z-scores (positive == worse) into one episode score.

    Args:
        z_by_metric: a dict of metric name -> z-score (NaNs are ignored).
            Metrics belonging to a disabled family are expected to simply be
            absent from this dict, not present with a NaN/zero value.
        weights: an optional dict of metric name -> weight, defaulting to
            1.0 for any metric not listed. The weighted mean is
            renormalized by the total weight of the metrics actually
            present in ``z_by_metric``, so a disabled family (whose
            metrics are absent) doesn't silently drag the score toward
            zero -- the remaining metrics' weights are rescaled to still
            average out over the metrics that are actually there.

    Returns:
        a tuple ``(overall_score, n_flags)``, where ``n_flags`` counts
        z-scores at or above :data:`WARN_Z` (unweighted)
    """
    weights = weights or {}
    items = [(name, z) for name, z in z_by_metric.items() if not np.isnan(z)]
    if not items:
        return 0.0, 0

    total_weight = sum(weights.get(name, 1.0) for name, _ in items)
    if total_weight <= 0:
        return 0.0, 0

    overall_score = sum(z * weights.get(name, 1.0) for name, z in items) / total_weight
    n_flags = sum(1 for _, z in items if z >= WARN_Z)
    return float(overall_score), n_flags


def severity(z_score):
    """Returns "fail"/"warn"/None for an oriented z-score (positive == worse)."""
    if np.isnan(z_score):
        return None
    if z_score >= FAIL_Z:
        return "fail"
    if z_score >= WARN_Z:
        return "warn"
    return None


def verdict(values_by_metric, stats_by_metric, higher_is_worse_fn):
    """Combines several metrics into one pass/warn/fail verdict.

    "fail" if any metric's z-score is fail-severity, else "warn" if any is
    warn-severity, else "pass". Metrics missing from ``stats_by_metric`` (or
    with a ``None``/NaN value) are skipped.

    Args:
        values_by_metric: a dict of metric name -> raw value
        stats_by_metric: a dict of metric name -> fitted ``(median, mad)``
            stats, as returned by :func:`fit`
        higher_is_worse_fn: a callable mapping a metric name to its polarity

    Returns:
        ``"pass"``, ``"warn"``, or ``"fail"``
    """
    return verdict_with_reason(values_by_metric, stats_by_metric, higher_is_worse_fn)[0]


def verdict_with_reason(values_by_metric, stats_by_metric, higher_is_worse_fn):
    """Like :func:`verdict`, but also names the metric that drove the verdict.

    Because :func:`severity` is monotonic in z, the highest-z metric is
    always one at the verdict's own severity level, so blaming it is
    consistent with the "any fail -> fail, else any warn -> warn" rule.

    Returns:
        a tuple ``(verdict, metric_name)``; ``metric_name`` is None when
        the verdict is ``"pass"``
    """
    worst_name, worst_z = None, None
    for name, value in values_by_metric.items():
        stats = stats_by_metric.get(name)
        if value is None or stats is None:
            continue

        z = zscore(value, stats, higher_is_worse_fn(name))
        if np.isnan(z):
            continue
        if worst_z is None or z > worst_z:
            worst_name, worst_z = name, z

    sev = severity(worst_z) if worst_z is not None else None
    if sev is None:
        return "pass", None
    return sev, worst_name
