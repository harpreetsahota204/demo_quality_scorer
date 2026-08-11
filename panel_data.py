"""Builds the Episode Quality panel's data payload.

The panel itself is a React component (see ``src/``); this module is its
data layer, called through the unlisted ``get_quality_panel_data`` operator.
Everything the frontend renders -- rows, histogram bins, verdicts, scatter
points, warn thresholds -- is computed here in one shot so the panel needs
exactly one backend call per refresh.
"""

import numpy as np

from .engine import normalize
from .engine.score import (
    HEALTH_METRIC_NAMES as HEALTH_METRICS,
    METRIC_WEIGHTS,
    MOTION_METRIC_NAMES as MOTION_METRICS,
    higher_is_worse,
    signal_metric_key,
)
from .operators import RUN_KEY

HISTOGRAM_BINS = 20

def build_panel_data(dataset, view):
    """Returns the full JSON-serializable payload for one panel refresh."""
    if "quality" not in dataset.get_field_schema():
        return {"scored": False}

    episode_stats = _episode_stats(dataset)

    rows = []
    config_versions = set()
    has_health = False
    for sample in view:
        q = sample.quality
        if q is None:
            continue
        config_versions.add(getattr(q, "config_version", None))
        health = getattr(q, "health", None)  # absent when family disabled
        has_health = has_health or health is not None
        verdict, reason = _health_verdict(health, episode_stats)
        by_signal = _motion_by_signal(q)
        rows.append(
            {
                "id": sample.id,
                "episode": _episode_label(sample),
                "overall_score": _num(q.overall_score),
                "n_flags": q.n_flags,
                **{name: _num(getattr(q, name, None)) for name in MOTION_METRICS},
                "by_signal": by_signal,
                "signal_scores": _signal_scores(by_signal, episode_stats),
                "worst_signal": _motion_worst_signal(q),
                "iforest_score": _num(getattr(q, "iforest_score", None)),
                "knn_dist": _num(getattr(q, "knn_dist", None)),
                "is_outlier": bool(q.is_outlier),
                "health_verdict": verdict,
                "health_reason": reason,
            }
        )
    # Worst-first is the panel's entire premise: the point of scoring is that a
    # reviewer reads down from the top and stops when episodes look fine.
    rows.sort(key=lambda r: -(r["overall_score"] or 0.0))

    signals = sorted({signal for r in rows for signal in r["by_signal"]})

    # Per-(metric, signal) warn thresholds, in each signal's own units.
    warn_thresholds = {}
    if episode_stats:
        for metric in MOTION_METRICS:
            per_signal = {}
            for signal in signals:
                key = signal_metric_key(metric, signal)
                stats = episode_stats.get(key)
                if stats:
                    per_signal[signal] = normalize.raw_value_at_z(
                        normalize.WARN_Z, stats, higher_is_worse(metric)
                    )
            if per_signal:
                warn_thresholds[metric] = per_signal

    histograms = {metric: _histogram(rows, signals, metric) for metric in MOTION_METRICS}

    verdict_counts = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
    for r in rows:
        verdict_counts[r["health_verdict"]] += 1

    return {
        "scored": True,
        "rows": rows,
        "signals": signals,
        "histograms": histograms,
        "warn_thresholds": warn_thresholds,
        "warn_z": normalize.WARN_Z,
        "fail_z": normalize.FAIL_Z,
        "verdict_counts": verdict_counts,
        "smoother_direction": {m: "left" if higher_is_worse(m) else "right" for m in MOTION_METRICS},
        "motion_scored": any(r[m] is not None for r in rows for m in MOTION_METRICS),
        "health_scored": has_health,
        "outliers_scored": any(r["iforest_score"] is not None for r in rows),
        # Every score here is relative to the corpus it was fit against, so
        # mixing formula versions in one ranking is meaningless. Reported so the
        # panel can say so rather than quietly sorting incomparable numbers.
        "config_version_mismatch": len(config_versions) > 1,
    }


def worst_interval_message(sample):
    """The deep-link toast text for one opened episode, or None.

    Tells the reviewer where to scrub, because FiftyOne can't seek the playhead
    programmatically -- opening the episode is as far as `OpenQualityEpisode`
    can get them.
    """
    intervals = getattr(sample, "quality_intervals", None)
    if not intervals:
        return None
    # Any `fail` interval, else the first one. Naming a single timestamp is the
    # job here; the timeline's temporal tags already show all of them.
    worst = max(intervals, key=lambda iv: iv.severity == "fail")
    return (
        f"Flag at {worst.start:.1f}-{worst.end:.1f}s "
        f"({worst.metric}, {worst.severity}) on {worst.channel} -- scrub to it."
    )


def _episode_stats(dataset):
    """The scoring run's cached episode-level normalization stats, or None.

    Without them the panel can still list every raw value, but nothing can be
    called good or bad -- thresholds and verdicts only exist relative to the
    corpus distribution these describe.
    """
    if not dataset.has_run(RUN_KEY):
        return None
    results = dataset.load_run_results(RUN_KEY)
    return results.norm_stats.get("episode") if results.norm_stats else None


def _health_verdict(health, episode_stats):
    """One pass/warn/fail for an episode's whole health family, plus the metric to blame.

    Five health numbers are too many for a table column a reviewer scans, and
    the useful question is binary anyway: is anything wrong with this
    recording. `"unknown"` (not `"pass"`) when the family was never scored --
    unmeasured is not the same as healthy.
    """
    if episode_stats is None or health is None:
        return "unknown", ""

    health_values = {f"health.{name}": getattr(health, name, None) for name in HEALTH_METRICS}
    verdict, worst = normalize.verdict_with_reason(health_values, episode_stats, higher_is_worse)
    reason = worst[len("health.") :] if worst else ""
    return verdict, reason


def _episode_label(sample):
    """A human-recognizable name for a row.

    Prefers a `task` field if the dataset has one, else the episode
    directory's name -- MCAP filenames are commonly non-distinct
    (``data.mcap``) while their parent directory is the episode id.
    """
    return getattr(sample, "task", None) or sample.filepath.rsplit("/", 2)[-2]


def _num(value):
    """None-safe float coercion (keeps the payload JSON-serializable)."""
    return float(value) if value is not None else None


def _motion_by_signal(quality_doc):
    """``{signal: {metric: value}}`` from a sample's quality document."""
    docs = getattr(quality_doc, "motion_by_signal", None) or ()
    return {
        doc.signal: {m: _num(getattr(doc, m, None)) for m in MOTION_METRICS}
        for doc in docs
    }


def _signal_scores(by_signal, episode_stats):
    """Per-signal motion score + flag count: ``{signal: {score, n_flags}}``.

    Same robust-z aggregation (and jerk_rms down-weighting) as the engine's
    `overall_score`, but restricted to one signal's motion metrics -- the
    ranking table sorts by this when the user isolates a signal in the
    legend. Motion-only by construction: health/outlier metrics aren't
    per-signal.
    """
    if not episode_stats:
        return {}

    scores = {}
    for signal, values in by_signal.items():
        z_by_metric = {}
        for metric in MOTION_METRICS:
            value = values.get(metric)
            key = signal_metric_key(metric, signal)
            stats = episode_stats.get(key)
            if value is None or stats is None:
                continue
            z = normalize.zscore(value, stats, higher_is_worse(metric))
            if not np.isnan(z):
                z_by_metric[metric] = z

        if z_by_metric:
            score, n_flags = normalize.aggregate(z_by_metric, weights=METRIC_WEIGHTS)
            scores[signal] = {"score": score, "n_flags": n_flags}
    return scores


def _motion_worst_signal(quality_doc):
    """``{metric: signal}`` attribution for the worst-of aggregation.

    Lets the panel say *which* arm was jerky, not just that the episode was.
    Values identify the exact selected signal that drove each metric.
    """
    doc = getattr(quality_doc, "motion_worst_signal", None)
    if doc is None:
        return {}
    return {m: getattr(doc, m, None) for m in MOTION_METRICS if getattr(doc, m, None)}


def _histogram(rows, signals, metric):
    """Multi-series histogram: shared bins across signals, one count per signal.

    Returns ``{"signals": [...], "bars": [{x, x0, x1, counts: {signal: n}}]}``.
    Bins are fit on the pooled values so signal series are directly
    overlayable; for unit-incompatible signals the shared axis is wide but
    still correct.
    """
    values_by_signal = {}
    for signal in signals:
        values = (r["by_signal"].get(signal, {}).get(metric) for r in rows)
        values_by_signal[signal] = [v for v in values if v is not None]

    pooled = [v for values in values_by_signal.values() for v in values]
    if not pooled:
        return {"signals": [], "bars": []}

    _, edges = np.histogram(np.asarray(pooled, dtype=np.float64), bins=HISTOGRAM_BINS)
    present = [signal for signal in signals if values_by_signal[signal]]
    counts_by_signal = {
        signal: np.histogram(np.asarray(values_by_signal[signal], dtype=np.float64), bins=edges)[0]
        for signal in present
    }
    return {
        "signals": present,
        "bars": [
            {
                "x": float((edges[i] + edges[i + 1]) / 2),
                "x0": float(edges[i]),
                "x1": float(edges[i + 1]),
                "counts": {signal: int(counts_by_signal[signal][i]) for signal in present},
            }
            for i in range(len(edges) - 1)
        ],
    }
