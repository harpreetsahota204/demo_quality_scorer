"""The `compute_episode_quality` operator: scores a view's episodes for quality."""

import json
from functools import lru_cache

import fiftyone.core.tags as fota
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np

from .engine import activity, motion, windowing
from .engine.decode import (
    channels_log_times,
    first_numeric_fields,
    missing_decoder_package,
)
from .engine.discovery import SCALAR_SIDECAR, TELEMETRY, discover_channels
from .engine.score import (
    CONFIG_VERSION,
    MotionSource,
    compute_raw_episode_metrics,
    finalize_batch,
)
from .write import build_temporal_tags, clear_temporal_tags, write_sample

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


def _signal_value(topic, group):
    """Stable JSON form value for one ``channel -> field group`` option."""
    return json.dumps([topic, group], separators=(",", ":"))


def _motion_sources(position_values, velocity_values):
    """Parses the form's two signal lists into engine motion sources."""
    sources = []
    seen = set()
    for kind, values in (("position", position_values), ("velocity", velocity_values)):
        for value in values or ():
            if value in seen:
                continue
            seen.add(value)
            topic, group = json.loads(value)
            sources.append(MotionSource(topic, group, kind))
    return sources


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
def _channel_field_groups(filepath, channel):
    fields = first_numeric_fields(filepath, channel)
    if not fields:
        return ()
    record = (0, 0, fields)
    return tuple(windowing.field_groups([record]))


def _auto_window_for_view(view, sources):
    """Resolves Auto windowing from every selected signal in the target view."""
    topics = {source.topic for source in sources}
    rates_hz, durations_s = [], []
    for sample in view:
        filepath = _local_path(sample)
        for times in channels_log_times(filepath, topics).values():
            times = np.asarray(times, dtype=np.int64)
            if len(times) < 2:
                continue
            gaps_s = np.diff(times).astype(np.float64) / 1e9
            positive = gaps_s[gaps_s > 0]
            if len(positive) == 0:
                continue
            rates_hz.append(1.0 / float(np.median(positive)))
            durations_s.append(float((times[-1] - times[0]) / 1e9))
    return windowing.resolve_auto_window(rates_hz, durations_s)


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
            # Both execution modes are always offered -- delegation is the
            # user's choice, never forced or blocked by view size. Scoring a
            # real corpus takes minutes, hence the delegated default.
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

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
        filepath = _local_path(view.first())
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

        # A channel is excluded from Motion when its first message decodes to
        # no numeric field groups. Missing decoders are reported separately;
        # without fields, the form cannot offer granular signal choices.
        # Failed imports are not cached by Python, so probe once per distinct
        # encoding rather than once per channel (dynamic=True re-resolves the
        # form on every interaction).
        package_by_encoding = {
            encoding: missing_decoder_package(encoding)
            for encoding in {c.message_encoding for c in telemetry}
        }

        missing_packages = {}  # pip package -> affected channel count
        motion_candidates = []
        for c in telemetry:
            package = package_by_encoding[c.message_encoding]
            if package is not None:
                missing_packages[package] = missing_packages.get(package, 0) + 1
            elif _channel_field_groups(filepath, c):
                motion_candidates.append(c)

        signal_choices = [
            (
                _signal_value(channel.topic, group),
                f"{channel.topic} \u2192 {group} ({channel.schema_name})",
            )
            for channel in motion_candidates
            for group in _channel_field_groups(filepath, channel)
        ]
        eligible_signals = {value for value, _ in signal_choices}
        has_motion = bool(signal_choices)

        if missing_packages:
            details = "; ".join(
                f"{count} channel(s) need {package}"
                for package, count in sorted(missing_packages.items())
            )
            inputs.view(
                "missing_decoders",
                types.Warning(
                    label=(
                        f"Missing MCAP decoders on this server: {details}. "
                        "Affected channels cannot expose field-group signal "
                        "choices until the package(s) are installed in the "
                        "server's Python environment."
                    )
                ),
            )

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
        position_signals = [
            value
            for value in (motion_cfg.get("position_signals") or [])
            if value in eligible_signals
        ]
        velocity_signals = [
            value
            for value in (motion_cfg.get("velocity_signals") or [])
            if value in eligible_signals and value not in position_signals
        ]
        motion_sources = _motion_sources(position_signals, velocity_signals)
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
                    "Smoothness per selected position or velocity signal; worst signal drives the score"
                    if has_motion
                    else "Skipped: no telemetry channel carries a numeric (speed-derivable) signal"
                ),
                default=has_motion,
            )
        if active_tab == "MOTION" and motion_enabled:
            section = _section("motion_cfg")

            # Metrics first (what to compute), signals second (where).
            section.view(
                "motion_metrics_header",
                types.Header(
                    label="Metrics",
                    description="Each is computed on every selected signal's speed profile",
                ),
            )
            _add_metric_checkboxes(section, MOTION_METRIC_ROWS)

            section.view("motion_channels_header", types.Header(label="Signals"))
            _channel_picker(
                section,
                "position_signals",
                signal_choices,
                position_signals,
                label="Position signals",
                description=(
                    "Coordinates or joint positions. Each selected signal is "
                    "differentiated once to produce speed."
                ),
            )
            _channel_picker(
                section,
                "velocity_signals",
                signal_choices,
                velocity_signals,
                label="Velocity signals",
                description=(
                    "Linear, angular, or joint velocities. Used directly and "
                    "preferred over position when both are available."
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
            section.bool(
                "auto_window",
                default=True,
                label="Choose window length automatically",
                description=(
                    "Uses all selected signals in the current view to target "
                    f"{windowing.AUTO_TARGET_SAMPLES} samples per window with a "
                    f"{windowing.AUTO_MIN_WINDOW_S:g}-second minimum."
                ),
            )
            if not motion_cfg.get("auto_window", True):
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
                label="Idle threshold (x moving speed)",
                description=(
                    "Fraction of the episode's typical speed while moving that counts "
                    "as idle. The moving-only reference stays above the noise floor in "
                    "mostly-idle episodes."
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
                source_keys = [source.key for source in motion_sources]
                # Genuinely wired: selected signals' motion
                # features are the columns the models consume. Health
                # features are episode-wide worst values (not per-signal), so
                # they always contribute and aren't selectable here.
                selected_outlier = [
                    key
                    for key in (outliers_cfg.get("outlier_signals") or [])
                    if key in source_keys
                ]
                _channel_picker(
                    section,
                    "outlier_signals",
                    [(key, key) for key in source_keys],
                    selected_outlier,
                    label="Which signals' motion features feed the models?",
                    description=(
                        "Leave empty to use all selected motion signals. Health "
                        "features are episode-wide and always contribute."
                    ),
                )
            else:
                section.view(
                    "outliers_info",
                    types.Notice(
                        label=(
                            "Models are fit on episode-wide health features only. "
                            "Select motion signals on the Motion tab to unlock "
                            "signal feature selection here."
                        )
                    ),
                )

        inputs.view(
            "validation",
            types.Notice(
                label=_build_validation_line(
                    has_motion=has_motion,
                    motion_enabled=motion_enabled,
                    n_motion_signals=len(motion_sources),
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

        overlap = _setting(motion_cfg, "overlap", windowing.OVERLAP)

        motion_enabled = ctx.params.get("motion_enabled", False)
        motion_sources = (
            _motion_sources(
                motion_cfg.get("position_signals"),
                motion_cfg.get("velocity_signals"),
            )
            if motion_enabled
            else []
        )
        motion_metrics = _selected_metrics(motion_cfg, MOTION_METRIC_ROWS)
        idle_alpha = _setting(motion_cfg, "idle_alpha", activity.IDLE_ALPHA_DEFAULT)
        jerk_cutoff_hz = _setting(motion_cfg, "jerk_cutoff_hz", motion.JERK_CUTOFF_HZ_DEFAULT)

        health_enabled = ctx.params.get("health_enabled", True)
        health_topics = set(health_cfg.get("health_channels") or []) if health_enabled else set()
        health_metrics = _selected_metrics(health_cfg, HEALTH_METRIC_ROWS)

        outliers_enabled = ctx.params.get("outliers_enabled", True)
        # Empty selection means all selected motion signals (engine: None).
        outlier_signals = outliers_cfg.get("outlier_signals") or None

        view = ctx.target_view()
        n = len(view)
        auto_window = bool(motion_cfg.get("auto_window", True))
        if auto_window and motion_sources:
            win_s, short_fraction = _auto_window_for_view(view, motion_sources)
        else:
            win_s = _setting(motion_cfg, "win_s", windowing.WINDOW_S)
            short_fraction = 0.0

        raw_by_id = {}
        for i, sample in enumerate(view):
            raw_by_id[sample.id] = compute_raw_episode_metrics(
                _local_path(sample),
                motion_sources=motion_sources,
                health_topics=health_topics,
                motion_metrics=motion_metrics,
                health_metrics=health_metrics,
                win_s=win_s,
                overlap=overlap,
                idle_alpha=idle_alpha,
                jerk_cutoff_hz=jerk_cutoff_hz,
            )
            # A yielded trigger reaches the App's progress bar but goes
            # nowhere in a delegated run; ctx.set_progress writes to the
            # delegated operation's record (visible on the Runs page) but
            # renders no bar in the App. One branch per mode, same message.
            if ctx.delegated:
                ctx.set_progress(progress=(i + 1) / n, label=f"Scored {i + 1}/{n}")
            else:
                yield ctx.trigger("set_progress", {"progress": (i + 1) / n, "label": f"Scored {i + 1}/{n}"})

        results, norm_stats = finalize_batch(
            raw_by_id, outliers_enabled=outliers_enabled, outlier_signals=outlier_signals
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
            motion_sources=motion_sources,
            motion_metrics=motion_metrics,
            health_topics=health_topics,
            health_metrics=health_metrics,
            outliers_enabled=outliers_enabled,
            outlier_signals=outlier_signals,
            win_s=win_s,
            auto_window=auto_window,
            short_fraction=short_fraction,
            overlap=overlap,
            idle_alpha=idle_alpha,
            jerk_cutoff_hz=jerk_cutoff_hz,
            norm_stats=norm_stats,
        )

        flagged = sum(1 for r in results.values() if r.n_flags > 0)
        # ctx.trigger raises "No executor available" in a delegated run --
        # there is no attached browser session to reload anyway; delegated
        # results land when the user next loads the dataset.
        if not ctx.delegated:
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
    n_motion_signals,
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
    elif n_motion_signals == 0:
        parts.append("Motion: no signals selected yet (Motion tab)")
    elif n_motion_metrics == 0:
        parts.append("Motion: no metrics selected -- family will compute nothing")
    else:
        parts.append(
            f"Motion: {n_motion_metrics} metric(s) on {n_motion_signals} signal(s), scored worst-of"
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
    motion_sources,
    motion_metrics,
    health_topics,
    health_metrics,
    outliers_enabled,
    outlier_signals,
    win_s,
    auto_window,
    short_fraction,
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
        motion_sources=[
            {"topic": source.topic, "group": source.group, "kind": source.kind}
            for source in motion_sources
        ],
        motion_metrics=list(motion_metrics),
        health_topics=sorted(health_topics),
        health_metrics=list(health_metrics),
        outliers_enabled=outliers_enabled,
        outlier_signals=sorted(outlier_signals) if outlier_signals else None,
        win_s=win_s,
        auto_window=auto_window,
        short_signal_fraction=short_fraction,
        overlap=overlap,
        idle_alpha=idle_alpha,
        jerk_cutoff_hz=jerk_cutoff_hz,
        config_version=CONFIG_VERSION,
    )
    dataset.register_run(RUN_KEY, cfg, overwrite=True)

    results = dataset.init_run_results(RUN_KEY)
    results.norm_stats = norm_stats
    dataset.save_run_results(RUN_KEY, results, overwrite=True)
