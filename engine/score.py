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
# Any metric not listed defaults to 1.0.
METRIC_WEIGHTS = {
    # Co-varies with sparc/ldlj/psd_lf_hf: all four summarize the same
    # underlying speed profile's roughness, so at equal weight "motion
    # roughness" implicitly outvotes every other family.
    "jerk_rms": 0.3,
    # Both are functions of every other metric in the feature matrix, so at
    # full weight the motion and health signal that produced them gets a
    # second vote -- the same double-counting argument as jerk_rms.
    "iforest_score": 0.5,
    "knn_dist": 0.5,
}

MIN_WINDOW_SAMPLES = 5

# Bumped whenever a formula change would shift already-written scores
# (idle_frac's relative threshold and jerk_rms's pre-filter both did, in the
# same change that introduced this constant -- the implicit "1" every score
# before this represents). Written to `quality.config_version` on every
# sample so the panel can refuse to blend runs computed under different
# formulas into one ranking.
# v3: motion metrics are computed per channel and normalized per
# (metric, channel), with top-level values taken from the worst channel.
# v4: a metric-math correctness pass. One-sided robust scale fit from each
# metric's bad tail (replacing a two-sided MAD with a fixed 1e-6 floor);
# psd_lf_hf returned as a log ratio; polarity-aware worst-window tails
# (`_worst`, replacing the always-p95 `_p95`); outlier models fit on oriented
# z-scores instead of raw mixed-unit columns; a moving-speed idle reference;
# gap-weighted dropout; MAD-based rate_cov; low-percentile desync; complete
# windows only; per-second speed profiles.
CONFIG_VERSION = 4

# Separator for channel-qualified metric keys ("sparc|/left-arm-state").
# Never written to FiftyOne field names (write.py restructures); only used
# for normalization stats and outlier feature-matrix keys.
CHANNEL_SEP = "|"

# Suffix for a motion metric's per-episode tail summary -- the worst single
# window, complementing the median over windows. Named for what it means
# rather than for the percentile that produces it, since which percentile
# that is depends on the metric's polarity (see `_tail_percentile`).
TAIL_SUFFIX = "_worst"
TAIL_PERCENTILE = 95


def channel_key(metric_name, topic):
    """The stats/feature-matrix key for one metric on one channel.

    Motion metrics are never pooled across channels -- a gripper's units and
    a shoulder joint's aren't comparable -- so every motion lookup is keyed by
    the pair, not the metric alone.
    """
    return f"{metric_name}{CHANNEL_SEP}{topic}"


@dataclass(frozen=True)
class WindowRecord:
    """One metric's value on one window of one channel's field group.

    The unit of interval flagging: `_finalize_intervals` z-scores each of
    these against window-level stats and merges the ones that clear a
    threshold into the spans that reach the multimodal timeline. `group` is
    carried separately from `metric` so a flag can name which
    sub-trajectory of a channel it came from ("pose.jerk_rms").
    """

    channel: str
    group: str
    metric: str
    start_s: float
    end_s: float
    value: float


@dataclass
class RawEpisodeMetrics:
    """One episode's metrics before the batch has been seen.

    Everything here is in raw metric units. Nothing is comparable, flagged or
    ranked yet: that all requires the corpus-wide distribution that only
    `finalize_batch` has. Kept as a separate type so an episode can be
    computed once (the expensive part -- it decodes MCAP) and normalized
    repeatedly against different batches.
    """

    scalars: dict = field(default_factory=dict)
    # {topic: {metric_name: value}} -- per-channel motion metrics (incl.
    # `_worst` tails, idle_frac, saturation_frac). Kept separate from
    # `scalars` because channel topics aren't valid FiftyOne field names and
    # because normalization is per (metric, channel).
    motion_by_channel: dict = field(default_factory=dict)
    window_records: list = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class EpisodeResult:
    """One episode's finalized, batch-relative scores -- what gets written to it.

    `overall_score` and `n_flags` are duplicated inside `scalars` as well;
    they're hoisted onto the dataclass because callers sort and summarize on
    them without caring about the rest.
    """

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
            channel+group's own *moving* speed (see
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
            # Per field group rather than per channel: a channel can pin one
            # sub-trajectory (a saturating wrist axis) while the rest of its
            # fields are healthy, and averaging the groups together first
            # would dilute that away before it is ever compared.
            for field_names in windowing.field_groups(records).values():
                _append_if_finite(health_values, "clipping_frac", health.clipping_frac(records, field_names))

    if "desync_ms" in health_metrics:
        # Desync is a property of a channel *pair*, so there is no per-channel
        # value to reduce -- every pair is measured and the episode reports the
        # worst one. Any single badly-skewed pair misaligns the episode's
        # image/action data, so the max is the number a reviewer needs.
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

                speed = motion.speed_profile(window.vectors, group, fs)
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

    # Reduce each channel's per-window values to two numbers: the median (what
    # the episode was typically like) and the worst window (whether it ever
    # went badly wrong). Both are needed -- a single bad grasp in a two-minute
    # episode barely moves the median, and an episode that is uniformly
    # mediocre never produces a standout worst window.
    motion_by_channel = {}
    for topic, motion_values in motion_values_by_topic.items():
        channel_scalars = {}
        for name in MOTION_METRIC_NAMES:
            if motion_values[name]:
                channel_scalars[name] = float(np.median(motion_values[name]))
                channel_scalars[name + TAIL_SUFFIX] = float(
                    np.percentile(motion_values[name], _tail_percentile(name))
                )
        for name in ("idle_frac", "saturation_frac"):
            if motion_values[name]:
                channel_scalars[name] = float(np.median(motion_values[name]))
        if channel_scalars:
            motion_by_channel[topic] = channel_scalars

    # Health metrics collapse to one episode-wide number per metric, median
    # across whatever contributed (channels, field groups, or channel pairs).
    # Median rather than max: a rig with 20 health channels would otherwise
    # have its score set by whichever single channel is flakiest, on every
    # episode, which ranks nothing.
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
        ``{"median": ..., "mad": ..., "scale_high": ..., "scale_low": ...}``
        dict (cacheable via a
        FiftyOne run, see :mod:`operators`). Motion entries in both dicts are keyed
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
        [[_raw_value(raw_by_id[sid], name) for name in metric_names] for sid in sample_ids],
        dtype=np.float64,
    ).reshape(len(sample_ids), len(metric_names))
    values_by_metric = {name: feature_matrix[:, i] for i, name in enumerate(metric_names)}

    # Normalization is fit *before* the outlier models, not after, so the
    # models can consume oriented z-scores rather than raw values.
    episode_stats = normalize.fit(values_by_metric)

    if outliers_enabled:
        iforest_scores, knn_dists = _fit_outliers(
            feature_matrix, metric_names, episode_stats, outlier_channels
        )
        values_by_metric["iforest_score"] = iforest_scores
        values_by_metric["knn_dist"] = knn_dists
        episode_stats.update(
            normalize.fit({"iforest_score": iforest_scores, "knn_dist": knn_dists})
        )

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
            if name + TAIL_SUFFIX in worst_scalars:
                scalars[name + TAIL_SUFFIX] = worst_scalars[name + TAIL_SUFFIX]

        # Activity scalars are context, not score: they never enter
        # `overall_score` (see AGGREGATE_METRICS), so there is no z-score to
        # take a worst-of over. Max across channels surfaces the most notable
        # value for a reviewer reading the number directly.
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
        # A boolean shortcut for the panel's filters, deliberately OR'd: the
        # two models catch different things (iforest sees odd summary stats,
        # kNN sees isolation in feature space), so either firing is worth a
        # look. The continuous scores remain the thing to rank by.
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
    """Turns an episode's flagged windows into the spans shown on the timeline.

    Windows are z-scored against *window-level* corpus stats, so "bad" means
    bad relative to every other window of the same metric on the same channel
    across the dataset -- not relative to this episode, which would flag
    something in every episode including the clean ones.

    Overlapping windows mean one real event flags 2+ consecutive windows, so
    contiguous runs are merged per series; without that, a single rough
    stretch would litter the timeline with near-duplicate tags.
    """
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
    """Merges touching/overlapping flagged windows of one series into single spans.

    A merged span reports the worst of what it absorbed (max value, and `fail`
    wins over `warn`), so collapsing the duplicates never softens a flag.
    Assumes `items` is sorted by start time.
    """
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
    """Polarity for any metric key, however it's qualified.

    Accepts the decorated forms that flow around this module -- channel-keyed
    (``"sparc|/left-arm"``) and tail (``"sparc_worst"``) -- so callers never
    have to strip a key before asking. Health metrics aren't listed in
    `motion.HIGHER_IS_WORSE`; they're all "more is worse", hence the default.
    """
    base = metric_name.split(CHANNEL_SEP, 1)[0]  # strip any "|<topic>" qualifier
    if base.endswith(TAIL_SUFFIX):
        base = base[: -len(TAIL_SUFFIX)]
    return motion.HIGHER_IS_WORSE.get(base, True)


def _tail_percentile(metric_name):
    """Which percentile of an episode's windows holds its *worst* window.

    p95 for a metric where higher is worse, p5 where lower is worse. Taking
    p95 unconditionally (as this used to) returns the single *smoothest*
    window for sparc, ldlj and psd_lf_hf -- the exact opposite of the "catch
    a bad stretch that the median washes out" job the tail summary exists to
    do.
    """
    return TAIL_PERCENTILE if higher_is_worse(metric_name) else 100 - TAIL_PERCENTILE


def _outlier_feature_columns(metric_names, outlier_channels):
    """Which feature columns feed the outlier models.

    Drops the :data:`TAIL_SUFFIX` columns: each is a percentile of the very
    same per-window distribution its median twin already summarizes, so
    keeping both counts the motion block twice in a Euclidean distance
    without adding information. Channel-qualified columns are subset by
    `outlier_channels` (None means all); unqualified health columns are
    episode-wide and always contribute.
    """
    allowed = None if outlier_channels is None else set(outlier_channels)
    columns = []
    for i, name in enumerate(metric_names):
        base, _, topic = name.partition(CHANNEL_SEP)
        if base.endswith(TAIL_SUFFIX):
            continue
        if allowed is not None and topic and topic not in allowed:
            continue
        columns.append(i)
    return columns


def _fit_outliers(feature_matrix, metric_names, episode_stats, outlier_channels):
    """Fits the outlier models on oriented z-scores rather than raw values.

    kNN distance is a plain Euclidean metric, so raw mixed-unit columns
    (jerk_rms in the thousands sitting next to dropout in [0, 1]) collapse it
    into a ranking on whichever column happens to have the widest raw
    spread. Robust z-scoring puts every column on a comparable scale first,
    and orienting them means "unusually bad" and "unusually good" at least
    share a sign convention. Isolation Forest is already invariant to
    per-feature rescaling -- its splits are drawn within each feature's own
    observed range -- but there's no reason to hand the two models different
    matrices.

    NaNs survive z-scoring and are imputed per column by
    :func:`.outliers.fit_and_score`.
    """
    n_samples = feature_matrix.shape[0]
    columns = _outlier_feature_columns(metric_names, outlier_channels)
    if not columns or n_samples == 0:
        return np.full(n_samples, np.nan), np.full(n_samples, np.nan)

    z_matrix = np.array(
        [
            [
                normalize.zscore(
                    feature_matrix[row, col],
                    episode_stats[metric_names[col]],
                    higher_is_worse(metric_names[col]),
                )
                for col in columns
            ]
            for row in range(n_samples)
        ]
    )
    return outliers.fit_and_score(z_matrix)


def _append_if_finite(values_by_metric, name, value):
    """Collects a metric value, dropping non-results.

    Metrics return NaN for "could not be computed here" (too few samples, a
    degenerate signal, only one clock). Dropping those keeps them out of the
    downstream median and out of the corpus stats fit, so an unmeasurable
    channel neither scores as 0 nor poisons the scale everything else is
    judged against.
    """
    if value is not None and not np.isnan(value):
        values_by_metric[name].append(value)
