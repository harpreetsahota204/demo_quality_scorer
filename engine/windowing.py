"""Uniform resampling and complete fixed-size windows for motion signals.

Each selected field group is resampled on its own median-rate grid; there is
no cross-channel alignment. Dropout-sized gaps remain invalid. Every emitted
window spans the same run-wide length because several motion metrics are
duration-dependent.
"""

import re
from dataclasses import dataclass

import numpy as np

from .health import DROPOUT_GAP_MULTIPLE

WINDOW_S = 2.0
OVERLAP = 0.5
AUTO_TARGET_SAMPLES = 100
AUTO_MIN_WINDOW_S = WINDOW_S
AUTO_RATE_PERCENTILE = 10
AUTO_WINDOW_INCREMENT_S = 0.5

_INDEXED_SUFFIX = re.compile(r"_\d+$")
# `w` is here so a quaternion groups as one 4-D orientation rather than
# splitting its scalar component off into a group of its own.
_AXIS_SUFFIX = re.compile(r"_(x|y|z|w|roll|pitch|yaw)$")


@dataclass(frozen=True)
class Window:
    """One time window of a channel's field-group vectors.

    ``vectors`` is an ``(n_samples, n_dims)`` array of the field-group's raw
    values within ``[start_s, end_s)``, on the signal's uniform time grid.
    """

    start_s: float
    end_s: float
    group: str
    vectors: np.ndarray


@dataclass(frozen=True)
class UniformSeries:
    """One field group resampled onto an evenly-spaced time grid."""

    times_s: np.ndarray
    vectors: np.ndarray
    fs: float


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
    """Builds one field-group's decoded ``(n_samples, n_dims)`` vector series."""
    return np.array([[fields.get(name, np.nan) for name in field_names] for _, _, fields in records])


def uniform_series(records, field_names, gap_multiple=DROPOUT_GAP_MULTIPLE):
    """Resamples a field group at its robust median message rate.

    Finite runs are linearly interpolated onto a uniform grid. Source gaps
    larger than ``gap_multiple`` expected intervals remain NaN so downstream
    motion metrics never mistake interpolation through a dropout for observed
    smooth motion.
    """
    if len(records) < 2:
        return None

    t0 = records[0][0]
    times_s = np.asarray([(r[0] - t0) / 1e9 for r in records], dtype=np.float64)
    gaps = np.diff(times_s)
    positive_gaps = gaps[gaps > 0]
    if len(positive_gaps) == 0:
        return None
    dt = float(np.median(positive_gaps))
    fs = 1.0 / dt
    grid = np.arange(0.0, times_s[-1] + 0.5 * dt, dt)
    source = np.asarray(group_vectors(records, field_names), dtype=np.float64)
    values = np.full((len(grid), source.shape[1]), np.nan, dtype=np.float64)

    for col in range(source.shape[1]):
        finite = np.isfinite(source[:, col])
        if finite.sum() < 2:
            continue
        source_times = times_s[finite]
        source_values = source[finite, col]
        source_times, unique_indices = np.unique(source_times, return_index=True)
        source_values = source_values[unique_indices]
        if len(source_times) < 2:
            continue
        values[:, col] = np.interp(grid, source_times, source_values)
        edge_tolerance = dt * 1e-9
        outside = (grid < source_times[0] - edge_tolerance) | (
            grid > source_times[-1] + edge_tolerance
        )
        values[outside, col] = np.nan
        long_gaps = np.diff(source_times) > gap_multiple * dt
        for left, right in zip(source_times[:-1][long_gaps], source_times[1:][long_gaps]):
            values[(grid > left) & (grid < right), col] = np.nan

    return UniformSeries(grid, values, fs)


def resolve_auto_window(rates_hz, durations_s):
    """Chooses one run-wide window from observed rates and episode lengths.

    The low rate percentile protects slower selected signals. A two-second
    floor preserves useful spectral resolution for ordinary high-rate
    telemetry; slower corpora receive a longer window to target roughly
    :data:`AUTO_TARGET_SAMPLES` observations. The value is rounded up to a
    half-second so it is stable and legible in cached run configuration.

    Returns ``(window_s, short_fraction)`` where ``short_fraction`` is the
    share of observed signal streams shorter than the resolved window.
    """
    rates = np.asarray(rates_hz, dtype=np.float64)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    durations = np.asarray(durations_s, dtype=np.float64)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if len(rates) == 0:
        return WINDOW_S, 0.0

    low_rate = float(np.percentile(rates, AUTO_RATE_PERCENTILE))
    candidate = max(AUTO_MIN_WINDOW_S, AUTO_TARGET_SAMPLES / low_rate)
    window_s = float(
        np.ceil(candidate / AUTO_WINDOW_INCREMENT_S) * AUTO_WINDOW_INCREMENT_S
    )
    short_fraction = float(np.mean(durations < window_s)) if len(durations) else 0.0
    return window_s, short_fraction


def windows_for_series(series, group, win_s=WINDOW_S, overlap=OVERLAP):
    """Builds complete windows from an already-uniform field-group series."""
    times_s = series.times_s
    vectors = series.vectors

    step_s = win_s * (1.0 - overlap)
    duration_s = times_s[-1]
    if duration_s <= 0 or step_s <= 0:
        return []

    if duration_s < win_s:
        return []

    # Only *complete* windows: a partial trailing window spans less time, and
    # ldlj's dimensionless jerk scales as roughly duration**4, so a
    # half-length window reads ~2.8 units smoother on identical motion. It
    # would never be flagged, and it would skew the channel's window-level
    # normalization stats for every other window too.
    windows = []
    start_s = 0.0
    while start_s + win_s <= duration_s:
        end_s = start_s + win_s
        mask = (times_s >= start_s) & (times_s < end_s)
        if mask.sum() >= 2 and np.isfinite(vectors[mask]).all():
            windows.append(Window(start_s, end_s, group, vectors[mask]))
        start_s += step_s

    return windows


def channel_duration_s(records):
    """Returns the span, in seconds, covered by a channel's decoded records."""
    if len(records) < 2:
        return 0.0
    return (records[-1][0] - records[0][0]) / 1e9
