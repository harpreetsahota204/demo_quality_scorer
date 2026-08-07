"""Activity metrics: idleness and pinned-at-extremes (saturation/clipping).

Saturation still operates on a window's raw field-group vectors directly.
Idleness now reuses :func:`motion.speed_profile`, so the "vel"-vs-"position"
naming heuristic used elsewhere in the engine also governs what "speed"
means here -- a velocity-named group's raw values ARE the speed, not
something to re-differentiate.
"""

import numpy as np

from .motion import lowpass_filtfilt, speed_profile

IDLE_ALPHA_DEFAULT = 0.05
IDLE_LOWPASS_CUTOFF_HZ = 10.0
PINNED_TOLERANCE = 1e-3


def episode_idle_threshold(vectors, group, fs, alpha=IDLE_ALPHA_DEFAULT):
    """Computes a per-episode idle-speed threshold, relative to that episode's own typical speed.

    Absolute thresholds don't generalize across unit scales (rad/s vs m/s
    vs normalized), so the threshold is a fraction of the channel+group's
    own median speed instead -- median, not peak, since peak is
    outlier-sensitive.

    Args:
        vectors: the FULL episode's ``(n_samples, n_dims)`` vectors for one
            channel+group (not a single window's) -- a stable reference
            requires the whole episode; a single window (possibly itself
            mostly idle) would give a near-zero, degenerate threshold
        group: the field-group name (passed through to `speed_profile`)
        fs: the channel's sampling rate, for the light low-pass
        alpha (0.05): fraction of median speed that counts as "idle"

    Returns:
        a threshold in the same units as the group's speed profile, or NaN
        if it can't be computed
    """
    speed = speed_profile(vectors, group)
    if len(speed) == 0 or np.isnan(fs) or fs <= 0:
        return np.nan
    filtered = lowpass_filtfilt(speed, IDLE_LOWPASS_CUTOFF_HZ, fs)
    return float(alpha * np.median(np.abs(filtered)))


def idle_frac(vectors, group, threshold, fs):
    """Fraction of a window's samples below an already-resolved idle-speed threshold.

    `threshold` is computed once per episode via `episode_idle_threshold`
    and passed in unchanged here, so every window of a group is judged
    against the same stable reference. The window's own speed is
    low-pass-filtered before comparing, matching the threshold's own
    filtering, to avoid spurious blips right at the threshold from noise
    near zero-crossings.
    """
    if len(vectors) < 2 or np.isnan(threshold) or np.isnan(fs) or fs <= 0:
        return np.nan
    speed = speed_profile(vectors, group)
    if len(speed) == 0:
        return np.nan
    filtered = lowpass_filtfilt(speed, IDLE_LOWPASS_CUTOFF_HZ, fs)
    return float(np.mean(filtered < threshold))


def pinned_fraction(vectors, tolerance=PINNED_TOLERANCE):
    """Fraction of samples pinned at their observed min/max (saturation/clipping heuristic).

    There's no generic way to know a channel's true actuator or sensor
    dtype limits, so this uses the data's own observed range as a proxy:
    samples within `tolerance` (relative to that range) of either extreme
    are considered pinned. Best-effort heuristic, not hardware-calibrated.
    """
    if len(vectors) == 0:
        return np.nan

    lo, hi = vectors.min(axis=0), vectors.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    near_lo = np.abs(vectors - lo) <= tolerance * span
    near_hi = np.abs(vectors - hi) <= tolerance * span
    return float(np.mean(near_lo | near_hi))
