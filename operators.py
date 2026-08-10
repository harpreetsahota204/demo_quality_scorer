"""The `compute_episode_quality` operator: scores a view's episodes for quality."""

from functools import lru_cache

import fiftyone.core.tags as fota
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .engine import activity, motion, windowing
from .engine.decode import has_numeric_signal
from .engine.discovery import SCALAR_SIDECAR, TELEMETRY, discover_channels
from .engine.score import (
    CONFIG_VERSION,
    compute_raw_episode_metrics,
    finalize_batch,
)
from .write import build_temporal_tags, clear_temporal_tags, write_sample

DELEGATION_THRESHOLD = 50
RUN_KEY = "demo_quality_scorer"

# Per-metric checkbox rows: (key, label, short description). Research notes
# follow the smoothness-metric literature: SPARC is the most validated and
# noise-robust (Balasubramanian 2015; stroke reaching review 2021); jerk-
# based metrics are noise/duration-sensitive.
MOTION_METRIC_ROWS = (
    (
        "sparc",
        "SPARC — spectral arc length",
        "Frequency-domain smoothness. The most validated and noise-robust "
        "of the four (recommended to keep on).",
    ),
    (
        "ldlj",
        "LDLJ — log dimensionless jerk",
        "Duration/amplitude-normalized jerk. Sensitive to sensor noise — "
        "consider deselecting on noisy telemetry.",
    ),
    (
        "jerk_rms",
        "Jerk RMS",
        "Raw jerk intensity (low-pass filtered first). Not scale-invariant; "
        "down-weighted 0.3x in the overall score.",
    ),
    (
        "psd_lf_hf",
        "PSD low/high ratio",
        "Share of motion energy in slow vs high-frequency bands (this "
        "plugin's own Welch band ratio).",
    ),
)

HEALTH_METRIC_ROWS = (
    (
        "dropout",
        "Dropout",
        "Estimated fraction of expected messages lost to gaps over 3x the "
        "expected interval (weighted by how many each gap swallowed).",
    ),
    (
        "rate_cov",
        "Rate stability",
        "Robust (MAD-based) coefficient of variation of message inter-arrival times.",
    ),
    ("clock_drift_ppm", "Clock drift", "log_time vs publish_time drift trend, in ppm (magnitude only)."),
    (
        "clipping_frac",
        "Clipping",
        "Fraction of samples pinned at their observed min/max (heuristic; "
        "constant and boolean fields are excluded, not flagged).",
    ),
    (
        "desync_ms",
        "Cross-channel desync",
        "Worst pairwise best-case timestamp offset; needs at least 2 channels.",
    ),
)


def _selected_metrics(cfg, rows):
    """The metric keys from ``rows`` whose checkboxes are on in a family's params."""
    return [key for key, _, _ in rows if cfg.get(f"metric_{key}", True)]


def _setting(cfg, name, default):
    """One numeric form value, falling back only when it is genuinely absent.

    Testing for None rather than falsiness, because 0 is a meaningful value for
    some of these knobs: an `overlap` of 0 means non-overlapping windows and an
    `idle_alpha` of 0 means nothing counts as idle. Treating those as unset
    would silently substitute the default for what the user asked for.
    """
    value = cfg.get(name)
    return default if value is None else value


def _add_metric_checkboxes(section, rows):
    """Renders a family's per-metric opt-outs, all on by default.

    Opt-out rather than opt-in: a user who doesn't know these metrics should
    get all of them and see which ones fire, and the descriptions carry the
    caveats that would justify unchecking one.
    """
    for key, label, description in rows:
        section.bool(
            f"metric_{key}",
            label=label,
            description=description,
            default=True,
            view=types.CheckboxView(),
        )


def _channel_picker(section, name, choices, selected, **field_kwargs):
    """A multi-select whose dropdown omits already-picked topics.

    Starts empty (an explicit pick beats a wall of pre-added pills); the
    choice list is rebuilt minus the current selection on every dynamic
    re-resolve, so a picked topic can't be added twice.

    Args:
        choices: a list of ``(topic, display_label)`` tuples
        selected: the currently-selected topics (from ``ctx.params``)
    """
    view = types.AutocompleteView()
    for topic, label in choices:
        if topic not in selected:
            view.add_choice(topic, label=label)
    section.list(name, types.String(), default=[], view=view, **field_kwargs)


def _local_path(sample):
    """The path to open ``sample``'s media from Python.

    On FiftyOne Enterprise, ``filepath`` can be a cloud URI (``gs://``,
    ``s3://``) that the engine's plain ``open()`` calls can't read.
    ``sample.local_path`` is the Enterprise SDK's answer: it downloads and
    caches the file locally on first access and returns that local path,
    or returns ``filepath`` unchanged if it's already local. That attribute
    doesn't exist in open-source FiftyOne, hence the ``getattr`` fallback --
    everything downstream of this (the whole ``engine/`` package) stays a
    plain local-file reader either way.
    """
    return getattr(sample, "local_path", None) or sample.filepath


# `dynamic=True` re-runs resolve_input on every form interaction, and
# uncached that meant re-reading the MCAP summary plus one message per
# telemetry channel on each toggle/keystroke. Episode files are immutable
# recordings, so caching per path is safe for a session's lifetime.
@lru_cache(maxsize=64)
def _discover_scorable(filepath):
    return tuple(c for c in discover_channels(filepath) if c.kind in (TELEMETRY, SCALAR_SIDECAR))


@lru_cache(maxsize=256)
def _channel_has_numeric_signal(filepath, channel):
    return has_numeric_signal(filepath, channel)


class ComputeEpisodeQuality(foo.Operator):
    """The plugin's one listed operator: scores a view's episodes end to end.

    A generator so a long run can report progress per episode instead of
    freezing the App, and delegable because scoring decodes every MCAP file in
    the view -- minutes of work on a real corpus.

    Scores are batch-relative by construction (see `engine.score`), so the
    unit of work is deliberately the whole target view: scoring a filtered
    subset produces z-scores relative to *that* subset, which is a different
    question than the one the same episode answers in a full-dataset run.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_episode_quality",
            label="Compute episode quality",
            description=(
                "Scores episodes for motion smoothness, sensor health, and "
                "outliers using the channels discovered on the current view."
            ),
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_delegation(self, ctx):
        """Defaults to delegated once the view is big enough to block the App."""
        return len(ctx.target_view()) > DELEGATION_THRESHOLD

    def resolve_input(self, ctx):
        """Builds the form from what the episodes actually contain.

        Nothing here is a fixed sensor list: the channel pickers and the
        Motion-family availability all come from discovery on a real file, so
        the same form works on a ROS bag and a protobuf recording. Warnings are
        soft -- the form never blocks Run, it explains what will be skipped
        (see `_build_validation_line`) and lets the user proceed.
        """
        inputs = types.Object()
        view = ctx.target_view()

        if len(view) == 0:
            inputs.view("empty_view", types.Warning(label="The current view has no samples."))
            return types.Property(inputs)

        # Scans only the view's first sample: discovery does not union
        # channels across episodes. Fine for a view recorded by a single
        # producer/schema, which is the common case, but a heterogeneous view
        # will only offer the channels its first episode happens to carry.
        sample0 = view.first()
        # TEMP DEBUG (remove once the Enterprise cloud-path issue is diagnosed):
        # ctx.log reaches the browser console; print() would not.
        ctx.log(
            "[demo-quality-scorer DEBUG] build=local_path-fix-v2 "
            f"sample.filepath={sample0.filepath!r} "
            f"has_local_path_attr={hasattr(sample0, 'local_path')} "
            f"local_path_value={getattr(sample0, 'local_path', '<no attr>')!r}"
        )
        filepath = _local_path(sample0)
        ctx.log(f"[demo-quality-scorer DEBUG] resolved filepath={filepath!r}")
        disc = list(_discover_scorable(filepath))
        if not disc:
            inputs.view(
                "no_channels",
                types.Warning(
                    label="No scorable channels found on the first sample "
                    "(need protobuf, ROS1/ROS2, or flat-JSON telemetry channels)."
                ),
            )
            return types.Property(inputs)

        telemetry = [c for c in disc if c.kind == TELEMETRY]
        motion_candidates = [c for c in telemetry if _channel_has_numeric_signal(filepath, c)]
        has_motion = bool(motion_candidates)

        # One tab per metric family. Only the active tab's fields are
        # rendered, but values persist in ctx.params across tab switches
        # (dynamic=True re-resolves the form on every change), and the
        # validation line below the tabs always reflects all families.
        # Tabs already separate the families visually, so sections are plain
        # vertical stacks (no outlined container) -- the nested object is
        # kept purely for param namespacing (ctx.params["motion_cfg"][...])
        def _section(name):
            prop = inputs.obj(name, view=types.GridView(orientation="vertical", gap=2))
            return prop.type

        # Selections are read from ctx.params up front (not inside the
        # render branches) so validation covers families on inactive tabs
        motion_cfg = ctx.params.get("motion_cfg") or {}
        health_cfg = ctx.params.get("health_cfg") or {}
        outliers_cfg = ctx.params.get("outliers_cfg") or {}

        motion_enabled = has_motion and ctx.params.get("motion_enabled", has_motion)
        eligible_motion = {c.topic for c in motion_candidates}
        motion_sources = [t for t in (motion_cfg.get("motion_sources") or []) if t in eligible_motion]
        motion_metrics = _selected_metrics(motion_cfg, MOTION_METRIC_ROWS)

        health_enabled = ctx.params.get("health_enabled", True)
        eligible_health = {c.topic for c in disc}
        health_channels = [t for t in (health_cfg.get("health_channels") or []) if t in eligible_health]
        health_metrics = _selected_metrics(health_cfg, HEALTH_METRIC_ROWS)

        outliers_enabled = ctx.params.get("outliers_enabled", True)

        tabs = types.TabsView()
        tabs.add_choice("MOTION", label="Motion")
        tabs.add_choice("HEALTH", label="Sensor health")
        tabs.add_choice("OUTLIERS", label="Outliers")
        inputs.enum("family_tab", tabs.values(), default="MOTION", view=tabs)
        active_tab = ctx.params.get("family_tab", "MOTION")

        # --- Motion smoothness -------------------------------------------
        if active_tab == "MOTION":
            inputs.bool(
                "motion_enabled",
                label="Motion smoothness",
                description=(
                    "Smoothness of the robot's motion, per channel, worst channel drives the score"
                    if has_motion
                    else "Skipped: no telemetry channel carries a numeric (speed-derivable) signal"
                ),
                default=has_motion,
            )
        if active_tab == "MOTION" and motion_enabled:
            section = _section("motion_cfg")

            # Metrics first (what to compute), channels second (where)
            section.view(
                "motion_metrics_header",
                types.Header(
                    label="Metrics",
                    description="Each is computed on every selected channel's speed profile",
                ),
            )
            _add_metric_checkboxes(section, MOTION_METRIC_ROWS)

            section.view("motion_channels_header", types.Header(label="Channels"))
            _channel_picker(
                section,
                "motion_sources",
                [(c.topic, f"{c.topic} ({c.schema_name})") for c in motion_candidates],
                motion_sources,
                required=True,
                label="Which channels carry motion?",
                description=(
                    "Each channel is scored independently and normalized "
                    "against its own dataset-wide stats; the worst channel "
                    "per metric drives the episode's score."
                ),
            )

            section.view(
                "motion_windowing_header",
                types.Header(
                    label="Windowing",
                    description=(
                        "Motion metrics are computed per window, then summarized per "
                        "episode (median + worst window). Windows only affect this family: "
                        "health uses full-episode timestamps, outliers use episode scalars."
                    ),
                ),
            )
            section.float(
                "win_s",
                default=windowing.WINDOW_S,
                label="Window length (s)",
                description="Shorter windows localize flags more precisely but get noisier.",
                min=0.5,
            )
            section.float(
                "overlap",
                default=windowing.OVERLAP,
                label="Window overlap",
                description="Fraction of overlap between consecutive windows (0 to 0.9).",
                min=0.0,
                max=0.9,
            )
            section.float(
                "idle_alpha",
                default=activity.IDLE_ALPHA_DEFAULT,
                label="Idle threshold (x p90 speed)",
                description=(
                    "Fraction of the episode's own 90th-percentile speed that counts as "
                    "idle. A high percentile, not the median: a mostly-idle episode's "
                    "median speed IS its idle floor."
                ),
                min=0.0,
            )
            if "jerk_rms" in motion_metrics:
                section.float(
                    "jerk_cutoff_hz",
                    default=motion.JERK_CUTOFF_HZ_DEFAULT,
                    label="Jerk pre-filter cutoff (Hz)",
                    description="Low-pass cutoff applied before differentiating for RMS jerk.",
                    min=0.1,
                )

        # --- Sensor health -------------------------------------------------
        if active_tab == "HEALTH":
            inputs.bool(
                "health_enabled",
                label="Sensor health",
                description="Timestamp- and value-level channel health, no motion needed",
                default=True,
            )
        if active_tab == "HEALTH" and health_enabled:
            section = _section("health_cfg")

            section.view(
                "health_metrics_header",
                types.Header(
                    label="Metrics",
                    description="Computed from raw message timestamps and values",
                ),
            )
            _add_metric_checkboxes(section, HEALTH_METRIC_ROWS)

            section.view("health_channels_header", types.Header(label="Channels"))
            _channel_picker(
                section,
                "health_channels",
                [(c.topic, f"{c.topic} ({c.schema_name})") for c in disc],
                health_channels,
                required=True,
                label="Which channels to check?",
            )

        # --- Outliers --------------------------------------------------
        if active_tab == "OUTLIERS":
            inputs.bool(
                "outliers_enabled",
                label="Outliers",
                description="Isolation forest + kNN manifold distance across the batch",
                default=True,
            )
        if active_tab == "OUTLIERS" and outliers_enabled:
            section = _section("outliers_cfg")
            if motion_sources:
                # Genuinely wired: selected channels' per-channel motion
                # features are the columns the models consume. Health
                # features are episode-wide medians (not per-channel), so
                # they always contribute and aren't selectable here.
                selected_outlier = [
                    t for t in (outliers_cfg.get("outlier_channels") or []) if t in motion_sources
                ]
                _channel_picker(
                    section,
                    "outlier_channels",
                    [(topic, topic) for topic in motion_sources],
                    selected_outlier,
                    label="Which channels' motion features feed the models?",
                    description=(
                        "Leave empty to use all selected motion channels. Health "
                        "features are episode-wide and always contribute."
                    ),
                )
            else:
                section.view(
                    "outliers_info",
                    types.Notice(
                        label=(
                            "Models are fit on episode-wide health features only. "
                            "Select motion channels on the Motion tab to unlock "
                            "per-channel feature selection here."
                        )
                    ),
                )

        inputs.view(
            "validation",
            types.Notice(
                label=_build_validation_line(
                    has_motion=has_motion,
                    motion_enabled=motion_enabled,
                    n_motion_channels=len(motion_sources),
                    n_motion_metrics=len(motion_metrics),
                    health_enabled=health_enabled,
                    n_health_channels=len(health_channels),
                    n_health_metrics=len(health_metrics),
                    outliers_enabled=outliers_enabled,
                )
            ),
        )

        return types.Property(inputs, view=types.View(label="Compute episode quality"))

    def execute(self, ctx):
        """Decodes every episode, then finalizes them as one batch.

        The two loops are not merge-able: normalization and the outlier models
        need every episode's raw metrics before any single episode's score
        exists, so the whole corpus is held in memory (raw metrics only -- the
        decoded telemetry is discarded per episode) between them.

        Yields progress during the first loop, since that's the expensive one.
        """
        # Per-family settings live in nested section objects (see resolve_input)
        motion_cfg = ctx.params.get("motion_cfg") or {}
        health_cfg = ctx.params.get("health_cfg") or {}
        outliers_cfg = ctx.params.get("outliers_cfg") or {}

        win_s = _setting(motion_cfg, "win_s", windowing.WINDOW_S)
        overlap = _setting(motion_cfg, "overlap", windowing.OVERLAP)

        motion_enabled = ctx.params.get("motion_enabled", False)
        motion_topics = (motion_cfg.get("motion_sources") or []) if motion_enabled else []
        motion_metrics = _selected_metrics(motion_cfg, MOTION_METRIC_ROWS)
        idle_alpha = _setting(motion_cfg, "idle_alpha", activity.IDLE_ALPHA_DEFAULT)
        jerk_cutoff_hz = _setting(motion_cfg, "jerk_cutoff_hz", motion.JERK_CUTOFF_HZ_DEFAULT)

        health_enabled = ctx.params.get("health_enabled", True)
        health_topics = set(health_cfg.get("health_channels") or []) if health_enabled else set()
        health_metrics = _selected_metrics(health_cfg, HEALTH_METRIC_ROWS)

        outliers_enabled = ctx.params.get("outliers_enabled", True)
        # Empty selection means "all selected motion channels" (engine: None)
        outlier_channels = outliers_cfg.get("outlier_channels") or None

        view = ctx.target_view()
        n = len(view)

        raw_by_id = {}
        for i, sample in enumerate(view):
            raw_by_id[sample.id] = compute_raw_episode_metrics(
                _local_path(sample),
                motion_topics=motion_topics,
                health_topics=health_topics,
                motion_metrics=motion_metrics,
                health_metrics=health_metrics,
                win_s=win_s,
                overlap=overlap,
                idle_alpha=idle_alpha,
                jerk_cutoff_hz=jerk_cutoff_hz,
            )
            yield ctx.trigger("set_progress", {"progress": (i + 1) / n, "label": f"Scored {i + 1}/{n}"})

        results, norm_stats = finalize_batch(
            raw_by_id, outliers_enabled=outliers_enabled, outlier_channels=outlier_channels
        )

        temporal_tags = []
        for sample in view.iter_samples(autosave=True):
            result = results[sample.id]
            write_sample(sample, result)
            temporal_tags.extend(build_temporal_tags(sample.id, result.intervals))

        # Clear-then-add (rather than add-only) so a re-run with a different
        # channel/family selection or window size doesn't leave stale tags
        # behind for intervals that no longer qualify.
        clear_temporal_tags(view)
        if temporal_tags:
            fota.add_temporal_tags(view, temporal_tags)

        _register_run(
            ctx.dataset,
            motion_topics=motion_topics,
            motion_metrics=motion_metrics,
            health_topics=health_topics,
            health_metrics=health_metrics,
            outliers_enabled=outliers_enabled,
            outlier_channels=outlier_channels,
            win_s=win_s,
            overlap=overlap,
            idle_alpha=idle_alpha,
            jerk_cutoff_hz=jerk_cutoff_hz,
            norm_stats=norm_stats,
        )

        flagged = sum(1 for r in results.values() if r.n_flags > 0)
        yield ctx.trigger("reload_dataset")
        yield {"scored": n, "flagged": flagged}

    def resolve_output(self, ctx):
        """A count only -- the panel, not a modal, is where results get read."""
        outputs = types.Object()
        result = ctx.results or {}
        outputs.str(
            "summary",
            label="Result",
            view=types.MarkdownView(),
            default=(
                f"**Scored {result.get('scored', 0)} episodes** -- "
                f"{result.get('flagged', 0)} flagged for review."
            ),
        )
        return types.Property(outputs)


def _build_validation_line(
    has_motion,
    motion_enabled,
    n_motion_channels,
    n_motion_metrics,
    health_enabled,
    n_health_channels,
    n_health_metrics,
    outliers_enabled,
):
    """Builds the soft-warn validation strip's text: never blocks Run, just explains skips."""
    parts = []

    if not has_motion:
        parts.append("Motion: skipped, no telemetry channel carries a numeric signal")
    elif not motion_enabled:
        parts.append("Motion: skipped by selection")
    elif n_motion_channels == 0:
        parts.append("Motion: no channels selected yet (Motion tab)")
    elif n_motion_metrics == 0:
        parts.append("Motion: no metrics selected -- family will compute nothing")
    else:
        parts.append(
            f"Motion: {n_motion_metrics} metric(s) on {n_motion_channels} channel(s), scored worst-of"
        )

    if not health_enabled:
        parts.append("Sensor health: skipped by selection")
    elif n_health_channels == 0:
        parts.append("Sensor health: no channels selected yet (Sensor health tab)")
    elif n_health_metrics == 0:
        parts.append("Sensor health: no metrics selected -- family will compute nothing")
    else:
        parts.append(f"Sensor health: {n_health_metrics} metric(s) on {n_health_channels} channel(s)")
        if n_health_channels < 2:
            parts.append("Desync: unavailable, needs >=2 health channels to compare")

    parts.append("Outliers: enabled" if outliers_enabled else "Outliers: skipped by selection")

    if not (motion_enabled or health_enabled or outliers_enabled):
        parts.append("Nothing selected -- nothing will be computed")

    return " · ".join(parts)


def _register_run(
    dataset,
    motion_topics,
    motion_metrics,
    health_topics,
    health_metrics,
    outliers_enabled,
    outlier_channels,
    win_s,
    overlap,
    idle_alpha,
    jerk_cutoff_hz,
    norm_stats,
):
    """Records what produced this run's scores, and the stats needed to reuse them.

    Two jobs. The config makes a run auditable and comparable: every knob that
    changes a score is stored alongside `config_version`, so the panel can tell
    that two episodes were scored under different formulas rather than silently
    ranking them together. The cached `norm_stats` are what let a later
    consumer z-score a new value against this run's corpus without refitting
    the whole batch.

    Overwrites on every run: this is the dataset's current scoring state, not a
    history. A previous run's stats no longer describe what's on the samples.
    """
    cfg = dataset.init_run(
        motion_topics=sorted(motion_topics),
        motion_metrics=list(motion_metrics),
        health_topics=sorted(health_topics),
        health_metrics=list(health_metrics),
        outliers_enabled=outliers_enabled,
        outlier_channels=sorted(outlier_channels) if outlier_channels else None,
        win_s=win_s,
        overlap=overlap,
        idle_alpha=idle_alpha,
        jerk_cutoff_hz=jerk_cutoff_hz,
        config_version=CONFIG_VERSION,
    )
    dataset.register_run(RUN_KEY, cfg, overwrite=True)

    results = dataset.init_run_results(RUN_KEY)
    results.norm_stats = norm_stats
    dataset.save_run_results(RUN_KEY, results, overwrite=True)
