"""Per-episode + corpus-wide orchestration of the metric engine.

Two phases, matching the fact that normalization and outlier detection both
need the whole batch before any single episode's score is final:

1. :func:`compute_raw_episode_metrics` -- one episode at a time, independent
   of every other episode.
2. :func:`finalize_batch` -- once, across every episode's raw output, to fit
   dataset-wide normalization + outlier models and produce final scores and
   flagged intervals.
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import activity, health, motion, normalize, outliers, windowing
from .decode import decode_channel
from .discovery import SCALAR_SIDECAR, TELEMETRY, discover_channels

MOTION_METRICS = (
    ("sparc", motion.sparc),
    ("ldlj", motion.ldlj),
    ("jerk_rms", motion.jerk_rms),
    ("psd_lf_hf", motion.psd_lf_hf),
)
MOTION_METRIC_NAMES = tuple(name for name, _ in MOTION_METRICS)
HEALTH_METRIC_NAMES = ("dropout", "desync_ms", "clock_drift_ppm", "rate_cov", "clipping_frac")

# Metrics that roll up into the worst-first `overall_score` / `n_flags`.
# idle_frac/saturation_frac are surfaced for context but aren't inherently
# "bad" at any level, so they're excluded from the aggregate.
AGGREGATE_METRICS = frozenset(
    set(MOTION_METRIC_NAMES)
    | {f"health.{name}" for name in HEALTH_METRIC_NAMES}
    | {"iforest_score", "knn_dist"}
)

# Per-metric weights for `overall_score` (normalize.aggregate renormalizes
# by the weight of whatever's actually present, so a metric absent because
# its family is disabled just isn't in the sum -- no special-casing here).
# Any metric not listed defaults to 1.0. jerk_rms is down-weighted since it
# tends to co-vary with sparc/ldlj/psd_lf_hf (all four summarize the same
# underlying speed profile's roughness), so an unweighted mean would let
# "motion roughness" implicitly outvote every other family.
METRIC_WEIGHTS = {"jerk_rms": 0.3}

MIN_WINDOW_SAMPLES = 5

# Bumped whenever a formula change would shift already-written scores
# (idle_frac's relative threshold and jerk_rms's pre-filter both did, in the
# same change that introduced this constant -- the implicit "1" every score
# before this represents). Written to `quality.config_version` on every
# sample so the panel can refuse to blend runs computed under different
# formulas into one ranking.
# v3: motion metrics are computed per channel and normalized per
# (metric, channel), with top-level values taken from the worst channel.
CONFIG_VERSION = 3

# Separator for channel-qualified metric keys ("sparc|/left-arm-state").
# Never written to FiftyOne field names (write.py restructures); only used
# for normalization stats and outlier feature-matrix keys.
CHANNEL_SEP = "|"


def channel_key(metric_name, topic):
    return f"{metric_name}{CHANNEL_SEP}{topic}"


@dataclass(frozen=True)
class WindowRecord:
    channel: str
    group: str
    metric: str
    start_s: float
    end_s: float
    value: float


@dataclass
class RawEpisodeMetrics:
    scalars: dict = field(default_factory=dict)
    # {topic: {metric_name: value}} -- per-channel motion metrics (incl.
    # `_p95`, idle_frac, saturation_frac). Kept separate from `scalars`
    # because channel topics aren't valid FiftyOne field names and because
    # normalization is per (metric, channel).
    motion_by_channel: dict = field(default_factory=dict)
    window_records: list = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class EpisodeResult:
    scalars: dict
    overall_score: float
    n_flags: int
    intervals: list
    duration_s: float
    config_version: int = CONFIG_VERSION
    # {topic: {metric_name: value}} -- raw per-channel motion metrics
    motion_by_channel: dict = field(default_factory=dict)
    # {metric_name: topic} -- which channel's z-score drove each top-level
    # motion value (worst-of aggregation)
    motion_worst_channel: dict = field(default_factory=dict)


def compute_raw_episode_metrics(
    filepath,
    motion_topics=(),
    health_topics=(),
    motion_metrics=MOTION_METRIC_NAMES,
    health_metrics=HEALTH_METRIC_NAMES,
    win_s=windowing.WINDOW_S,
    overlap=windowing.OVERLAP,
    idle_alpha=activity.IDLE_ALPHA_DEFAULT,
    jerk_cutoff_hz=10.0,
):
    """Computes one episode's raw (not-yet-normalized) quality metrics.

    Motion and health are scored over independently-selectable channel sets
    (a family the operator's form let the user uncheck contributes nothing
    here, rather than being filtered out later): pass `motion_topics=()`
    and/or `health_topics=()` to skip either family entirely. Within a
    family, `motion_metrics`/`health_metrics` select which individual
    metrics to compute -- a deselected metric is simply absent from the
    output (and `normalize.aggregate` renormalizes over what's present).

    Args:
        filepath: path to the episode's MCAP file
        motion_topics: the channel topics to run motion-smoothness metrics
            (SPARC/LDLJ/jerk RMS/PSD ratio/idle/saturation) on. Each topic
            is scored independently (per-channel values land in
            ``RawEpisodeMetrics.motion_by_channel``); empty to skip the
            Motion family
        health_topics: the channel topics to run sensor-health metrics
            (dropout, rate stability, clock drift, clipping, pairwise
            desync) on; empty to skip the Health family
        motion_metrics: which motion metrics to compute, a subset of
            :data:`MOTION_METRIC_NAMES`
        health_metrics: which health metrics to compute, a subset of
            :data:`HEALTH_METRIC_NAMES`
        win_s: window length, in seconds (motion windows only; health
            metrics use the episode's full message stream)
        overlap: fractional overlap between consecutive windows
        idle_alpha (0.05): idle-speed threshold, as a fraction of each
            channel+group's own median speed (see
            `activity.episode_idle_threshold`)
        jerk_cutoff_hz (10.0): low-pass cutoff, in Hz, for `motion.jerk_rms`'s
            pre-filter

    Returns:
        a :class:`RawEpisodeMetrics`
    """
    motion_topics = set(motion_topics)
    health_topics = set(health_topics)
    motion_metrics = set(motion_metrics)
    health_metrics = set(health_metrics)
    needed_topics = health_topics | motion_topics

    channels = {
        c.topic: c
        for c in discover_channels(filepath)
        if c.topic in needed_topics and c.kind in (TELEMETRY, SCALAR_SIDECAR)
    }
    channel_records = {topic: decode_channel(filepath, c) for topic, c in channels.items()}
    channel_records = {topic: recs for topic, recs in channel_records.items() if recs}
    duration_s = max((windowing.channel_duration_s(recs) for recs in channel_records.values()), default=0.0)

    motion_values_by_topic = {topic: defaultdict(list) for topic in motion_topics}
    health_values = defaultdict(list)
    window_records = []

    for topic in health_topics:
        records = channel_records.get(topic)
        if not records:
            continue

        if {"dropout", "rate_cov"} & health_metrics:
            _, dropout, rate_cov = health.rate_stats(records)
            if "dropout" in health_metrics:
                _append_if_finite(health_values, "dropout", dropout)
            if "rate_cov" in health_metrics:
                _append_if_finite(health_values, "rate_cov", rate_cov)

        if "clock_drift_ppm" in health_metrics:
            drift = health.clock_drift_ppm(records)
            _append_if_finite(health_values, "clock_drift_ppm", abs(drift) if not np.isnan(drift) else drift)

        if "clipping_frac" in health_metrics:
            for field_names in windowing.field_groups(records).values():
                _append_if_finite(health_values, "clipping_frac", health.clipping_frac(records, field_names))

    if "desync_ms" in health_metrics:
        health_records = [channel_records[t] for t in health_topics if t in channel_records]
        desyncs = [health.desync_ms(a, b) for a, b in itertools.combinations(health_records, 2)]
        desyncs = [d for d in desyncs if not np.isnan(d)]
        if desyncs:
            health_values["desync_ms"].append(max(desyncs))

    for motion_topic in motion_topics:
        records = channel_records.get(motion_topic)
        if not records:
            continue

        motion_values = motion_values_by_topic[motion_topic]
        expected_dt, _, _ = health.rate_stats(records)
        fs = 1.0 / expected_dt if expected_dt and expected_dt > 0 else np.nan

        for group, field_names in windowing.field_groups(records).items():
            full_vectors = windowing.group_vectors(records, field_names)
            idle_threshold = activity.episode_idle_threshold(full_vectors, group, fs, alpha=idle_alpha)

            for window in windowing.windows_for_group(records, group, field_names, win_s, overlap):
                if window.vectors.shape[0] < MIN_WINDOW_SAMPLES or np.isnan(fs):
                    continue

                _append_if_finite(
                    motion_values, "idle_frac", activity.idle_frac(window.vectors, group, idle_threshold, fs)
                )
                _append_if_finite(motion_values, "saturation_frac", activity.pinned_fraction(window.vectors))

                speed = motion.speed_profile(window.vectors, group)
                for metric_name, metric_fn in MOTION_METRICS:
                    if metric_name not in motion_metrics:
                        continue
                    value = (
                        metric_fn(speed, fs, jerk_cutoff_hz) if metric_name == "jerk_rms" else metric_fn(speed, fs)
                    )
                    if np.isnan(value):
                        continue
                    motion_values[metric_name].append(value)
                    window_records.append(
                        WindowRecord(motion_topic, group, metric_name, window.start_s, window.end_s, value)
                    )

    motion_by_channel = {}
    for topic, motion_values in motion_values_by_topic.items():
        channel_scalars = {}
        for name in MOTION_METRIC_NAMES:
            if motion_values[name]:
                channel_scalars[name] = float(np.median(motion_values[name]))
                channel_scalars[f"{name}_p95"] = float(np.percentile(motion_values[name], 95))
        for name in ("idle_frac", "saturation_frac"):
            if motion_values[name]:
                channel_scalars[name] = float(np.median(motion_values[name]))
        if channel_scalars:
            motion_by_channel[topic] = channel_scalars

    scalars = {}
    for name, values in health_values.items():
        if values:
            scalars[f"health.{name}"] = float(np.median(values))

    return RawEpisodeMetrics(
        scalars=scalars,
        motion_by_channel=motion_by_channel,
        window_records=window_records,
        duration_s=duration_s,
    )


def finalize_batch(raw_by_id, outliers_enabled=True, outlier_channels=None):
    """Normalizes a batch of episodes' raw metrics and finalizes their scores.

    Fits dataset-wide robust-z stats and the outlier models across every
    episode in ``raw_by_id`` at once, so every episode in the batch is
    scored on the same scale (and re-running later with more episodes will
    shift everyone's scores together, by design).

    Args:
        raw_by_id: a dict of sample id -> :class:`RawEpisodeMetrics`
        outliers_enabled (True): whether to fit the Isolation Forest/kNN
            outlier models at all. When False, `iforest_score`/`knn_dist`
            are omitted entirely (not merely excluded from the aggregate),
            since fitting them costs a pass over the whole batch that
            nobody asked for.
        outlier_channels (None): which channels' per-channel *motion*
            features feed the outlier models; ``None`` means all of them.
            Health features are episode-wide medians (not per-channel), so
            they always contribute. Only affects the models' feature
            matrix -- normalization stats and z-scores are unaffected.

    Returns:
        a tuple ``(results, norm_stats)`` where ``results`` is a dict of
        sample id -> :class:`EpisodeResult` and ``norm_stats`` is
        ``{"episode": ..., "window": ...}``, each a per-metric
        ``{"median": ..., "mad": ...}`` dict (cacheable via a FiftyOne run,
        see :mod:`operators`). Motion entries in both dicts are keyed
        per (metric, channel) as ``"<metric>|<topic>"`` -- different
        channels can carry different units (rad/s vs m/s), so pooling them
        into one stats fit would be meaningless. Episode- and window-level
        values are normalized separately because raw per-window values are
        naturally far more spread out than episode-level
        (median-of-windows) values; z-scoring one against stats fit on the
        other would flag almost every above-average window as a severe
        outlier.

    Motion metrics aggregate **worst-of across channels**: each channel is
    z-scored against its own (metric, channel) stats, and the highest
    (worst) z per metric drives the top-level value, `overall_score`, and
    `n_flags`. Rationale: this scorer is triage, and averaging (as e.g.
    RINSE does for bimanual filtering) would let one smooth arm mask a
    jerky one -- exactly the flag a reviewer needs to see.
    """
    sample_ids = list(raw_by_id)

    # Feature keys: plain health names + channel-qualified motion names.
    # Per-channel motion features feed the outlier models directly.
    metric_names = sorted(
        {name for raw in raw_by_id.values() for name in raw.scalars}
        | {
            channel_key(name, topic)
            for raw in raw_by_id.values()
            for topic, channel_scalars in raw.motion_by_channel.items()
            for name in channel_scalars
        }
    )

    def _raw_value(raw, key):
        base, _, topic = key.partition(CHANNEL_SEP)
        if topic:
            return raw.motion_by_channel.get(topic, {}).get(base, np.nan)
        return raw.scalars.get(key, np.nan)

    feature_matrix = np.array(
        [[_raw_value(raw_by_id[sid], name) for name in metric_names] for sid in sample_ids]
    )
    values_by_metric = {name: feature_matrix[:, i] for i, name in enumerate(metric_names)}

    if outliers_enabled:
        outlier_matrix = feature_matrix
        if outlier_channels is not None:
            allowed = set(outlier_channels)
            cols = [
                i
                for i, name in enumerate(metric_names)
                if CHANNEL_SEP not in name or name.split(CHANNEL_SEP, 1)[1] in allowed
            ]
            outlier_matrix = feature_matrix[:, cols]

        if outlier_matrix.shape[1] == 0:
            iforest_scores = knn_dists = np.full(len(sample_ids), np.nan)
        else:
            iforest_scores, knn_dists = outliers.fit_and_score(outlier_matrix)
        values_by_metric["iforest_score"] = iforest_scores
        values_by_metric["knn_dist"] = knn_dists
    episode_stats = normalize.fit(values_by_metric)

    window_values_by_metric = defaultdict(list)
    for raw in raw_by_id.values():
        for record in raw.window_records:
            window_values_by_metric[channel_key(record.metric, record.channel)].append(record.value)
    window_stats = normalize.fit(window_values_by_metric)

    results = {}
    for row, sid in enumerate(sample_ids):
        raw = raw_by_id[sid]
        scalars = dict(raw.scalars)
        if outliers_enabled:
            scalars["iforest_score"] = float(values_by_metric["iforest_score"][row])
            scalars["knn_dist"] = float(values_by_metric["knn_dist"][row])

        # Health + outlier metrics: one z per plain metric name
        z_by_metric = {
            name: normalize.zscore(scalars[name], episode_stats[name], higher_is_worse(name))
            for name in scalars
            if name in episode_stats
        }

        # Motion metrics: z per (metric, channel), worst-of across channels
        motion_worst_channel = {}
        for name in MOTION_METRIC_NAMES:
            worst_z, worst_topic = None, None
            for topic, channel_scalars in raw.motion_by_channel.items():
                if name not in channel_scalars:
                    continue
                key = channel_key(name, topic)
                if key not in episode_stats:
                    continue
                z = normalize.zscore(channel_scalars[name], episode_stats[key], higher_is_worse(name))
                if np.isnan(z):
                    continue
                if worst_z is None or z > worst_z:
                    worst_z, worst_topic = z, topic

            if worst_topic is None:
                continue
            z_by_metric[name] = worst_z
            motion_worst_channel[name] = worst_topic
            worst_scalars = raw.motion_by_channel[worst_topic]
            scalars[name] = worst_scalars[name]
            if f"{name}_p95" in worst_scalars:
                scalars[f"{name}_p95"] = worst_scalars[f"{name}_p95"]

        # Context-only activity scalars: max across channels (most notable)
        for name in ("idle_frac", "saturation_frac"):
            values = [cs[name] for cs in raw.motion_by_channel.values() if name in cs]
            if values:
                scalars[name] = max(values)

        overall_score, n_flags = normalize.aggregate(
            {name: z for name, z in z_by_metric.items() if name in AGGREGATE_METRICS},
            weights=METRIC_WEIGHTS,
        )
        scalars["overall_score"] = overall_score
        scalars["n_flags"] = n_flags
        scalars["is_outlier"] = bool(outliers_enabled) and (
            z_by_metric.get("iforest_score", 0) >= normalize.WARN_Z
            or z_by_metric.get("knn_dist", 0) >= normalize.WARN_Z
        )

        intervals = _finalize_intervals(raw.window_records, window_stats)
        results[sid] = EpisodeResult(
            scalars=scalars,
            overall_score=overall_score,
            n_flags=n_flags,
            intervals=intervals,
            duration_s=raw.duration_s,
            config_version=CONFIG_VERSION,
            motion_by_channel=raw.motion_by_channel,
            motion_worst_channel=motion_worst_channel,
        )

    return results, {"episode": episode_stats, "window": window_stats}


def _series_key(item):
    """Groups window records by the metric series they belong to (a channel+group+metric)."""
    record, _sev = item
    return (record.channel, record.group, record.metric)


def _finalize_intervals(window_records, norm_stats):
    flagged = []
    for record in window_records:
        stats = norm_stats.get(channel_key(record.metric, record.channel))
        if stats is None:
            continue
        z = normalize.zscore(record.value, stats, higher_is_worse(record.metric))
        sev = normalize.severity(z)
        if sev is not None:
            flagged.append((record, sev))

    flagged.sort(key=lambda item: _series_key(item) + (item[0].start_s,))

    intervals = []
    for (channel, group, metric), group_items in itertools.groupby(flagged, key=_series_key):
        for merged in _merge_contiguous(list(group_items)):
            intervals.append(
                {
                    "channel": channel,
                    "metric": f"{group}.{metric}",
                    "start": merged["start"],
                    "end": merged["end"],
                    "value": merged["value"],
                    "severity": merged["severity"],
                }
            )
    return intervals


def _merge_contiguous(items):
    merged = []
    for record, sev in items:
        if merged and record.start_s <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], record.end_s)
            merged[-1]["value"] = max(merged[-1]["value"], record.value)
            merged[-1]["severity"] = "fail" if "fail" in (merged[-1]["severity"], sev) else "warn"
        else:
            merged.append({"start": record.start_s, "end": record.end_s, "value": record.value, "severity": sev})
    return merged


def higher_is_worse(metric_name):
    base = metric_name.split(CHANNEL_SEP, 1)[0]  # strip any "|<topic>" qualifier
    base = base[: -len("_p95")] if base.endswith("_p95") else base
    return motion.HIGHER_IS_WORSE.get(base, True)


def _append_if_finite(values_by_metric, name, value):
    if value is not None and not np.isnan(value):
        values_by_metric[name].append(value)
