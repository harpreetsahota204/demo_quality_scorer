"""Windowing: turns a decoded channel's scalar records into fixed-size windows.

2s windows, 50% overlap, on the channel's own nanosecond timebase (no
cross-channel resampling here -- each channel is windowed against its own
timestamps, since telemetry channels in a single episode are commonly
sampled on independent clocks).
"""

import re
from dataclasses import dataclass

import numpy as np

WINDOW_S = 2.0
OVERLAP = 0.5

_INDEXED_SUFFIX = re.compile(r"_\d+$")
_AXIS_SUFFIX = re.compile(r"_(x|y|z|roll|pitch|yaw)$")


@dataclass(frozen=True)
class Window:
    """One time window of a channel's field-group vectors.

    ``vectors`` is an ``(n_samples, n_dims)`` array of the field-group's raw
    values within ``[start_s, end_s)``, at the channel's native sample times.
    """

    start_s: float
    end_s: float
    group: str
    vectors: np.ndarray


def field_groups(records):
    """Groups a channel's flat field names by base name (strips index/axis suffixes).

    E.g. ``position_0..5`` -> group ``"position"``; ``pose_x/y/z`` -> group
    ``"pose"``. This recovers the multi-dimensional sub-trajectories that
    :func:`.decode.decode_channel` flattened, without assuming any specific
    field names.

    Args:
        records: the decoded records for one channel, as returned by
            :func:`.decode.decode_channel`

    Returns:
        a dict of group name -> sorted list of full field names
    """
    groups = {}
    for _, _, fields in records:
        for name in fields:
            base = _AXIS_SUFFIX.sub("", _INDEXED_SUFFIX.sub("", name))
            groups.setdefault(base, set()).add(name)

    return {group: sorted(names) for group, names in groups.items()}


def group_vectors(records, field_names):
    """Builds one field-group's full-episode ``(n_samples, n_dims)`` vector series.

    This is the same array :func:`windows_for_group` slices into windows
    internally, exposed separately for callers (e.g. the per-episode
    idle-speed threshold) that need a stable, whole-episode reference
    rather than a single window's slice of it.
    """
    return np.array([[fields.get(name, np.nan) for name in field_names] for _, _, fields in records])


def windows_for_group(records, group, field_names, win_s=WINDOW_S, overlap=OVERLAP):
    """Builds overlapping fixed-size windows of one field-group's vector series.

    Args:
        records: the decoded records for one channel
        group: the field-group name (from :func:`field_groups`)
        field_names: the ordered field names making up this group's vector
        win_s (2.0): window length, in seconds
        overlap (0.5): fractional overlap between consecutive windows

    Returns:
        a list of :class:`Window`
    """
    if not records:
        return []

    t0 = records[0][0]
    times_s = np.array([(log_time - t0) / 1e9 for log_time, _, _ in records])
    vectors = group_vectors(records, field_names)

    step_s = win_s * (1.0 - overlap)
    duration_s = times_s[-1]
    if duration_s <= 0:
        return []

    windows = []
    start_s = 0.0
    while start_s < duration_s:
        end_s = start_s + win_s
        mask = (times_s >= start_s) & (times_s < end_s)
        if mask.sum() >= 2:
            windows.append(Window(start_s, end_s, group, vectors[mask]))
        start_s += step_s

    return windows


def channel_duration_s(records):
    """Returns the span, in seconds, covered by a channel's decoded records."""
    if len(records) < 2:
        return 0.0
    return (records[-1][0] - records[0][0]) / 1e9
