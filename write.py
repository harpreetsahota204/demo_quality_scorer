"""Writes engine.score.EpisodeResult scores onto FiftyOne samples.

Kept separate from the engine package so the metric engine itself has no
FiftyOne dependency and stays independently unit-testable.

Flagged intervals are written as multimodal *temporal tags*
(`fiftyone.core.tags`), not a `TemporalDetections` label field: the
multimodal player's timeline builds its tracks exclusively from the
temporal-tags collection (confirmed against
`app/packages/multimodal/src/adapters/mcap/react/use-mcap-temporal-tags.ts`
and `fiftyone/multimodal/server/routes.py`) and has no code path that reads
arbitrary label fields -- `TemporalDetections`-on-timeline rendering exists
only in the separate classic-video package, gated to `media_type="video"`.
A `TemporalDetections` field on a multimodal sample is fully valid data (it
shows up in the sidebar and as grid-overlay text) but will never appear on
the player's timeline, so it can't be used to "scrub to the flag."

Note: `fiftyone.core.tags` is undocumented/internal (hidden from Sphinx via
`@hide_from_docs` on the public `Collection.temporal_tags` accessor) -- it
works in the installed version this plugin was built against, but carries
none of the stability guarantees of fiftyone's public API.
"""

import fiftyone as fo
import fiftyone.core.tags as fota

# Every temporal tag this plugin writes carries this anchor, so a later run
# can find-and-clear exactly its own tags (see clear_temporal_tags) without
# ever touching a tag a person drew by hand on the multimodal timeline.
TEMPORAL_TAG_ANCHOR = "demo_quality_scorer"


def build_quality_document(
    scalars,
    config_version,
    motion_by_signal=None,
    motion_sources=None,
    motion_worst_signal=None,
):
    """Builds the ``quality`` field payload for one sample from its raw scalars.

    Args:
        scalars: an :class:`engine.score.EpisodeResult`'s ``scalars`` dict
            (flat, with ``"health.<name>"`` keys for nested health fields).
            Motion values here are already the worst selected signal's (see
            ``engine.score.finalize_batch``)
        config_version: the :class:`engine.score.EpisodeResult`'s
            ``config_version`` -- provenance, not a metric, so it's kept
            out of ``scalars`` (and out of any z-scoring/aggregation) and
            written directly as ``quality.config_version``
        motion_by_signal (None): ``{signal_key: {metric: value}}``, written as
            embedded signal documents
        motion_sources (None): source provenance keyed by signal
        motion_worst_signal (None): ``{metric: signal_key}``

    Returns:
        a :class:`fiftyone.core.odm.embedded_document.DynamicEmbeddedDocument`
    """
    top_level, health = {}, {}
    for key, value in scalars.items():
        if key.startswith("health."):
            health[key[len("health.") :]] = value
        else:
            top_level[key] = value

    if health:
        top_level["health"] = fo.DynamicEmbeddedDocument(**health)
    if motion_by_signal:
        docs = []
        for signal, values in sorted(motion_by_signal.items()):
            source = (motion_sources or {}).get(signal)
            provenance = (
                {
                    "signal": signal,
                    "channel": source.topic,
                    "group": source.group,
                    "kind": source.kind,
                }
                if source is not None
                else {"channel": signal}
            )
            docs.append(fo.DynamicEmbeddedDocument(**provenance, **values))
        top_level["motion_by_signal"] = docs
    if motion_worst_signal:
        top_level["motion_worst_signal"] = fo.DynamicEmbeddedDocument(**motion_worst_signal)
    top_level["config_version"] = config_version
    return fo.DynamicEmbeddedDocument(**top_level)


def build_quality_intervals(intervals):
    """Builds the ``quality_intervals`` field payload for one sample.

    The same flags `build_temporal_tags` puts on the timeline, stored a second
    time as an ordinary sample field. Temporal tags render but aren't readable
    as sample data, so this is what lets the panel and view expressions
    actually query what was flagged.
    """
    return [fo.DynamicEmbeddedDocument(**interval) for interval in intervals]


def build_temporal_tags(sample_id, intervals):
    """Builds this episode's flagged intervals as multimodal temporal tags.

    Args:
        sample_id: the sample's ID
        intervals: an :class:`engine.score.EpisodeResult`'s ``intervals``
            list

    Returns:
        a list of :class:`fiftyone.core.tags.TemporalTag`, one per interval,
        all carrying :data:`TEMPORAL_TAG_ANCHOR`
    """
    tags = []
    for interval in intervals:
        start_ns = round(interval["start"] * 1e9)
        end_ns = round(interval["end"] * 1e9)
        if end_ns <= start_ns:
            end_ns = start_ns + 1  # temporal tags require a strict start < end

        # Temporal tags are unique on (sample, start, end, tag): the channel
        # must be part of `tag`, not left implicit, or two different
        # channels flagging the same metric in the same rounded window
        # collide and silently deduplicate down to one tag.
        tags.append(
            fota.TemporalTag(
                sample_id=sample_id,
                start=start_ns,
                end=end_ns,
                tag=f"{interval['channel']}:{interval['metric']}:{interval['severity']}",
                anchor=TEMPORAL_TAG_ANCHOR,
            )
        )
    return tags


def clear_temporal_tags(dataset_or_view):
    """Deletes this plugin's own previously-written temporal tags in the given scope.

    Safe to call before every re-scoring run: only tags carrying
    :data:`TEMPORAL_TAG_ANCHOR` are touched, so hand-drawn temporal tags a
    person created in the App are never affected.
    """
    dataset_or_view.temporal_tags.delete(
        filter=fota.TemporalTagFilter(anchors=TEMPORAL_TAG_ANCHOR)
    )


def write_sample(sample, result):
    """Writes one episode's :class:`engine.score.EpisodeResult` onto its sample.

    Does not call ``sample.save()`` or write temporal tags; callers own the
    save (e.g. batching via ``iter_samples(autosave=True)``) and should batch
    :func:`build_temporal_tags` + :func:`clear_temporal_tags` +
    ``fiftyone.core.tags.add_temporal_tags`` once across the whole run
    instead of per sample.
    """
    sample["quality"] = build_quality_document(
        result.scalars,
        result.config_version,
        motion_by_signal=result.motion_by_signal,
        motion_sources=result.motion_sources,
        motion_worst_signal=result.motion_worst_signal,
    )
    sample["quality_intervals"] = build_quality_intervals(result.intervals)
