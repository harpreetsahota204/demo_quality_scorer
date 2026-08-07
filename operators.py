"""The `compute_episode_quality` operator: scores a view's episodes for quality."""

from functools import lru_cache

import fiftyone.core.tags as fota
import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import debug
from .engine import activity, windowing
from .engine.decode import has_numeric_signal
from .engine.discovery import SCALAR_SIDECAR, TELEMETRY, discover_channels
from .engine.score import CONFIG_VERSION, compute_raw_episode_metrics, finalize_batch
from .write import build_temporal_tags, clear_temporal_tags, write_sample

DELEGATION_THRESHOLD = 50
RUN_KEY = "demo_quality_scorer"


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

    def _debug(self, ctx, event, **fields):
        debug.log(ctx, "compute_episode_quality", event, **fields)

    def resolve_delegation(self, ctx):
        return len(ctx.target_view()) > DELEGATION_THRESHOLD

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = ctx.target_view()

        if len(view) == 0:
            inputs.view("empty_view", types.Warning(label="The current view has no samples."))
            return types.Property(inputs)

        # Scans only the view's first sample, same as the as-built flat-list
        # form this replaces -- discovery does not union channels across
        # multiple episodes. Fine for a single MCAP producer/schema, which
        # is the common case; noted here rather than adding new scan
        # machinery in this change (out of scope).
        filepath = view.first().filepath
        disc = list(_discover_scorable(filepath))
        self._debug(
            ctx,
            "resolve_input.channels",
            first_sample_filepath=filepath,
            discovered=[(c.topic, c.kind, c.schema_name) for c in disc],
        )
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
        self._debug(
            ctx,
            "resolve_input.motion_candidates",
            candidates=[c.topic for c in motion_candidates],
            has_motion=has_motion,
        )

        # --- Motion smoothness -------------------------------------------
        inputs.bool(
            "motion_enabled",
            label="Motion smoothness",
            description=(
                "SPARC, LDLJ, jerk RMS, PSD band ratio, idle/saturation fraction"
                if has_motion
                else "Skipped: no telemetry channel carries a numeric (speed-derivable) signal"
            ),
            default=has_motion,
        )
        motion_enabled = has_motion and ctx.params.get("motion_enabled", has_motion)
        motion_sources = []
        if motion_enabled:
            mc = types.AutocompleteView()
            for c in motion_candidates:
                mc.add_choice(c.topic, label=f"{c.topic} ({c.schema_name})")
            inputs.list(
                "motion_sources",
                types.String(),
                default=[c.topic for c in motion_candidates],
                required=True,
                label="Which channels carry motion?",
                description=(
                    "Each channel is scored independently and normalized "
                    "against its own dataset-wide stats; the worst channel "
                    "per metric drives the episode's score."
                ),
                view=mc,
            )
            motion_sources = ctx.params.get("motion_sources") or [c.topic for c in motion_candidates]
            inputs.float(
                "idle_alpha",
                default=activity.IDLE_ALPHA_DEFAULT,
                label="Idle threshold (x median speed)",
                description="Fraction of the episode's own median speed that counts as idle.",
                min=0.0,
            )
            inputs.float(
                "jerk_cutoff_hz",
                default=10.0,
                label="Jerk pre-filter cutoff (Hz)",
                description="Low-pass cutoff applied before differentiating for RMS jerk.",
                min=0.1,
            )

        # --- Sensor health -------------------------------------------------
        inputs.bool("health_enabled", label="Sensor health", default=True)
        health_enabled = ctx.params.get("health_enabled", True)
        health_channels = []
        if health_enabled:
            ch = types.AutocompleteView()
            for c in disc:
                ch.add_choice(c.topic, label=f"{c.topic} ({c.schema_name})")
            inputs.list(
                "health_channels",
                types.String(),
                default=[c.topic for c in disc],
                required=True,
                label="Which channels to check?",
                description="Dropout, desync, clock drift, rate stability, clipping.",
                view=ch,
            )
            health_channels = ctx.params.get("health_channels") or [c.topic for c in disc]

        # --- Outliers --------------------------------------------------
        inputs.bool("outliers_enabled", label="Outliers", default=True)
        outliers_enabled = ctx.params.get("outliers_enabled", True)
        if outliers_enabled:
            # No channel picker here on purpose: an earlier version offered
            # one, but both models are fit on the batch's already-computed
            # quality scalars, so the selection changed nothing. A live
            # control that does nothing is worse than an honest notice;
            # re-add the picker if/when per-channel features get wired up.
            inputs.view(
                "outliers_info",
                types.Notice(
                    label=(
                        "Isolation forest + kNN manifold distance, fit on every "
                        "computed quality scalar (motion + health) across the batch."
                    )
                ),
            )

        inputs.float("win_s", default=windowing.WINDOW_S, label="Window length (s)", min=0.5)
        inputs.float(
            "overlap",
            default=windowing.OVERLAP,
            label="Window overlap",
            description="Fraction of overlap between consecutive windows.",
            view=types.SliderView(min=0.0, max=0.9),
        )

        inputs.view(
            "validation",
            types.Notice(
                label=_build_validation_line(
                    has_motion,
                    motion_enabled,
                    len(motion_sources),
                    health_enabled,
                    len(health_channels),
                    outliers_enabled,
                )
            ),
        )

        return types.Property(inputs, view=types.View(label="Compute episode quality"))

    def execute(self, ctx):
        win_s = ctx.params.get("win_s") or windowing.WINDOW_S
        overlap = ctx.params.get("overlap")
        overlap = windowing.OVERLAP if overlap is None else overlap

        motion_enabled = ctx.params.get("motion_enabled", False)
        motion_topics = (ctx.params.get("motion_sources") or []) if motion_enabled else []
        idle_alpha = ctx.params.get("idle_alpha") or activity.IDLE_ALPHA_DEFAULT
        jerk_cutoff_hz = ctx.params.get("jerk_cutoff_hz") or 10.0

        health_enabled = ctx.params.get("health_enabled", True)
        health_topics = set(ctx.params.get("health_channels") or []) if health_enabled else set()

        outliers_enabled = ctx.params.get("outliers_enabled", True)

        view = ctx.target_view()
        n = len(view)
        self._debug(
            ctx,
            "execute.start",
            motion_topics=sorted(motion_topics),
            health_topics=sorted(health_topics),
            outliers_enabled=outliers_enabled,
            win_s=win_s,
            overlap=overlap,
            n_samples=n,
            delegated=ctx.delegated,
        )

        raw_by_id = {}
        for i, sample in enumerate(view):
            raw_by_id[sample.id] = compute_raw_episode_metrics(
                sample.filepath,
                motion_topics=motion_topics,
                health_topics=health_topics,
                win_s=win_s,
                overlap=overlap,
                idle_alpha=idle_alpha,
                jerk_cutoff_hz=jerk_cutoff_hz,
            )
            self._debug(
                ctx,
                "execute.scored_sample",
                i=i,
                sample_id=sample.id,
                filepath=sample.filepath,
                n_window_records=len(raw_by_id[sample.id].window_records),
            )
            yield ctx.trigger("set_progress", {"progress": (i + 1) / n, "label": f"Scored {i + 1}/{n}"})

        results, norm_stats = finalize_batch(raw_by_id, outliers_enabled=outliers_enabled)
        self._debug(
            ctx,
            "execute.finalized",
            episode_metrics=sorted(norm_stats["episode"]),
            window_metrics=sorted(norm_stats["window"]),
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
            health_topics=health_topics,
            outliers_enabled=outliers_enabled,
            win_s=win_s,
            overlap=overlap,
            idle_alpha=idle_alpha,
            jerk_cutoff_hz=jerk_cutoff_hz,
            norm_stats=norm_stats,
        )

        flagged = sum(1 for r in results.values() if r.n_flags > 0)
        self._debug(
            ctx,
            "execute.done",
            scored=n,
            flagged=flagged,
            run_key=RUN_KEY,
            n_temporal_tags=len(temporal_tags),
        )

        yield ctx.trigger("reload_dataset")
        yield {"scored": n, "flagged": flagged}

    def resolve_output(self, ctx):
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
    has_motion, motion_enabled, n_motion_channels, health_enabled, n_health_channels, outliers_enabled
):
    """Builds the soft-warn validation strip's text: never blocks Run, just explains skips."""
    parts = []

    if not has_motion:
        parts.append("Motion: skipped, no telemetry channel carries a numeric signal")
    elif motion_enabled:
        parts.append(f"Motion: enabled on {n_motion_channels} channel(s), scored worst-of")
    else:
        parts.append("Motion: skipped by selection")

    if health_enabled:
        parts.append(f"Sensor health: enabled on {n_health_channels} channel(s)")
        if n_health_channels < 2:
            parts.append("Desync: unavailable, needs >=2 health channels to compare")
    else:
        parts.append("Sensor health: skipped by selection")

    parts.append("Outliers: enabled" if outliers_enabled else "Outliers: skipped by selection")

    if not (motion_enabled or health_enabled or outliers_enabled):
        parts.append("Nothing selected -- nothing will be computed")

    return " · ".join(parts)


def _register_run(
    dataset,
    motion_topics,
    health_topics,
    outliers_enabled,
    win_s,
    overlap,
    idle_alpha,
    jerk_cutoff_hz,
    norm_stats,
):
    cfg = dataset.init_run(
        motion_topics=sorted(motion_topics),
        health_topics=sorted(health_topics),
        outliers_enabled=outliers_enabled,
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
