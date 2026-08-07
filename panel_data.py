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
    MOTION_METRIC_NAMES as MOTION_METRICS,
    channel_key,
    higher_is_worse,
)
from .operators import RUN_KEY

HISTOGRAM_BINS = 20

# Pseudo-channel used for scores written before per-channel motion metrics
# existed (config_version < 3): those samples only carry top-level values.
LEGACY_CHANNEL = "all"


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
        rows.append(
            {
                "id": sample.id,
                "episode": _episode_label(sample),
                "overall_score": _num(q.overall_score),
                "n_flags": q.n_flags,
                **{name: _num(getattr(q, name, None)) for name in MOTION_METRICS},
                "by_channel": _motion_by_channel(q),
                "worst_channel": _motion_worst_channel(q),
                "iforest_score": _num(getattr(q, "iforest_score", None)),
                "knn_dist": _num(getattr(q, "knn_dist", None)),
                "is_outlier": bool(q.is_outlier),
                "health_verdict": verdict,
                "health_reason": reason,
            }
        )
    rows.sort(key=lambda r: -(r["overall_score"] or 0.0))

    channels = sorted({channel for r in rows for channel in r["by_channel"]})

    # Per-(metric, channel) warn thresholds, in each channel's own units.
    # Falls back to legacy plain-key stats for pre-v3 runs.
    warn_thresholds = {}
    if episode_stats:
        for metric in MOTION_METRICS:
            per_channel = {}
            for channel in channels:
                key = metric if channel == LEGACY_CHANNEL else channel_key(metric, channel)
                stats = episode_stats.get(key)
                if stats:
                    per_channel[channel] = normalize.raw_value_at_z(
                        normalize.WARN_Z, stats, higher_is_worse(metric)
                    )
            if per_channel:
                warn_thresholds[metric] = per_channel

    histograms = {metric: _histogram(rows, channels, metric) for metric in MOTION_METRICS}

    verdict_counts = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
    for r in rows:
        verdict_counts[r["health_verdict"]] += 1

    return {
        "scored": True,
        "rows": rows,
        "channels": channels,
        "histograms": histograms,
        "warn_thresholds": warn_thresholds,
        "verdict_counts": verdict_counts,
        "smoother_direction": {m: "left" if higher_is_worse(m) else "right" for m in MOTION_METRICS},
        "motion_scored": any(r[m] is not None for r in rows for m in MOTION_METRICS),
        "health_scored": has_health,
        "outliers_scored": any(r["iforest_score"] is not None for r in rows),
        "config_version_mismatch": len(config_versions) > 1,
    }


def worst_interval_message(sample):
    """The deep-link toast text for one opened episode, or None."""
    intervals = getattr(sample, "quality_intervals", None)
    if not intervals:
        return None
    worst = max(intervals, key=lambda iv: iv.severity == "fail")
    return (
        f"Flag at {worst.start:.1f}-{worst.end:.1f}s "
        f"({worst.metric}, {worst.severity}) on {worst.channel} -- scrub to it."
    )


def _episode_stats(dataset):
    if not dataset.has_run(RUN_KEY):
        return None
    results = dataset.load_run_results(RUN_KEY)
    return results.norm_stats.get("episode") if results.norm_stats else None


def _health_verdict(health, episode_stats):
    if episode_stats is None or health is None:
        return "unknown", ""

    health_values = {f"health.{name}": getattr(health, name, None) for name in HEALTH_METRICS}
    verdict, worst = normalize.verdict_with_reason(health_values, episode_stats, higher_is_worse)
    reason = worst[len("health.") :] if worst else ""
    return verdict, reason


def _episode_label(sample):
    return getattr(sample, "task", None) or sample.filepath.rsplit("/", 2)[-2]


def _num(value):
    """None-safe float coercion (keeps the payload JSON-serializable)."""
    return float(value) if value is not None else None


def _motion_by_channel(quality_doc):
    """``{channel: {metric: value}}`` from a sample's quality doc.

    Pre-v3 scores have no ``motion_by_channel`` list; their top-level values
    are surfaced under the :data:`LEGACY_CHANNEL` pseudo-channel so the
    panel renders one series either way.
    """
    docs = getattr(quality_doc, "motion_by_channel", None)
    if docs:
        return {
            doc.channel: {m: _num(getattr(doc, m, None)) for m in MOTION_METRICS}
            for doc in docs
        }

    legacy = {m: _num(getattr(quality_doc, m, None)) for m in MOTION_METRICS}
    if any(v is not None for v in legacy.values()):
        return {LEGACY_CHANNEL: legacy}
    return {}


def _motion_worst_channel(quality_doc):
    doc = getattr(quality_doc, "motion_worst_channel", None)
    if doc is None:
        return {}
    return {m: getattr(doc, m, None) for m in MOTION_METRICS if getattr(doc, m, None)}


def _histogram(rows, channels, metric):
    """Multi-series histogram: shared bins across channels, one count per channel.

    Returns ``{"channels": [...], "bars": [{x, x0, x1, counts: {channel: n}}]}``.
    Bins are fit on the pooled values so channel series are directly
    overlayable; for unit-incompatible channels the shared axis is wide but
    still correct.
    """
    values_by_channel = {}
    for channel in channels:
        values = (r["by_channel"].get(channel, {}).get(metric) for r in rows)
        values_by_channel[channel] = [v for v in values if v is not None]

    pooled = [v for values in values_by_channel.values() for v in values]
    if not pooled:
        return {"channels": [], "bars": []}

    _, edges = np.histogram(np.asarray(pooled, dtype=np.float64), bins=HISTOGRAM_BINS)
    present = [c for c in channels if values_by_channel[c]]
    counts_by_channel = {
        channel: np.histogram(np.asarray(values_by_channel[channel], dtype=np.float64), bins=edges)[0]
        for channel in present
    }
    return {
        "channels": present,
        "bars": [
            {
                "x": float((edges[i] + edges[i + 1]) / 2),
                "x0": float(edges[i]),
                "x1": float(edges[i + 1]),
                "counts": {channel: int(counts_by_channel[channel][i]) for channel in present},
            }
            for i in range(len(edges) - 1)
        ],
    }
