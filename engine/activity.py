"""Activity metrics: idleness and pinned-at-extremes (saturation/clipping).

Saturation operates on a window's raw field-group vectors directly.
Idleness reuses :func:`motion.speed_profile`, so the "vel"-vs-"position"
naming heuristic used elsewhere in the engine also governs what "speed"
means here -- a velocity-named group's raw values ARE the speed, not
something to re-differentiate.
"""

import numpy as np

from .motion import lowpass_filtfilt, speed_profile

IDLE_ALPHA_DEFAULT = 0.05
IDLE_LOWPASS_CUTOFF_HZ = 10.0
# Provisional split point for separating "moving" from "idle" samples when
# estimating an episode's reference speed -- see _moving_speed_reference.
IDLE_REFERENCE_PERCENTILE = 90
PINNED_TOLERANCE = 1e-3
# A dimension needs at least this many distinct values before "pinned at an
# extreme" carries any information -- see pinned_fraction.
PINNED_MIN_DISTINCT = 3


def _moving_speed_reference(filtered):
    """An episode's typical speed *while it's moving*.

    No single percentile can serve as this reference: whatever percentile you
    pick, an episode idle for more than that fraction of its length puts the
    percentile itself back down in the noise floor. So the high percentile is
    only a provisional split, and the reference is the median of everything
    above half of it -- one refinement step, enough to pull the reference out
    of the idle mass for any episode that spends a few percent of its length
    moving. (An episode that is essentially 100% idle has no moving speed to
    find and still degenerates; nothing here can fix that, and such an
    episode reads as an outlier on every other metric anyway.)
    """
    provisional = float(np.percentile(filtered, IDLE_REFERENCE_PERCENTILE))
    moving = filtered[filtered > provisional / 2.0]
    if len(moving) == 0:
        return provisional
    return float(np.median(moving))


def episode_idle_threshold(vectors, group, fs, alpha=IDLE_ALPHA_DEFAULT):
    """Computes a per-episode idle-speed threshold, relative to that episode's own moving speed.

    Absolute thresholds don't generalize across unit scales (rad/s vs m/s vs
    normalized), so the threshold is a fraction of the channel+group's own
    moving speed instead.

    That reference is emphatically not the episode's median speed, which is
    what this used to use: a median-relative threshold inverts on exactly the
    episodes the metric exists to catch. Once an episode is more than half
    idle its median speed *is* the idle floor, so the threshold collapses to
    a fraction of the sensor noise and `idle_frac` reports ~0 -- a 60%-idle
    episode scored *lower* than a 20%-idle one. See
    :func:`_moving_speed_reference`.

    Args:
        vectors: the FULL episode's ``(n_samples, n_dims)`` vectors for one
            channel+group (not a single window's) -- a stable reference
            requires the whole episode; a single window's speed distribution
            shifts with whatever that window happens to contain
        group: the field-group name (passed through to `speed_profile`)
        fs: the channel's sampling rate, for the light low-pass
        alpha (0.05): fraction of the reference speed that counts as "idle"

    Returns:
        a threshold in the same units as the group's speed profile, or NaN
        if it can't be computed
    """
    if np.isnan(fs) or fs <= 0:
        return np.nan
    speed = speed_profile(vectors, group, fs)
    if len(speed) == 0:
        return np.nan
    filtered = np.abs(lowpass_filtfilt(speed, IDLE_LOWPASS_CUTOFF_HZ, fs))
    return float(alpha * _moving_speed_reference(filtered))


def idle_frac(vectors, group, threshold, fs):
    """Fraction of a window's samples below an already-resolved idle-speed threshold.

    `threshold` is computed once per episode via `episode_idle_threshold`
    and passed in unchanged here, so every window of a group is judged
    against the same stable reference. The window's own speed is
    low-pass-filtered and rectified exactly as the threshold's own reference
    was -- filtered, to avoid spurious blips right at the threshold from
    noise near zero-crossings; rectified, because filtfilt can push an
    otherwise non-negative speed slightly below zero there.
    """
    if len(vectors) < 2 or np.isnan(threshold) or np.isnan(fs) or fs <= 0:
        return np.nan
    speed = speed_profile(vectors, group, fs)
    if len(speed) == 0:
        return np.nan
    filtered = np.abs(lowpass_filtfilt(speed, IDLE_LOWPASS_CUTOFF_HZ, fs))
    return float(np.mean(filtered < threshold))


def pinned_fraction(vectors, tolerance=PINNED_TOLERANCE, min_distinct=PINNED_MIN_DISTINCT):
    """Fraction of samples pinned at their observed min/max (saturation/clipping heuristic).

    There's no generic way to know a channel's true actuator or sensor dtype
    limits, so this uses the data's own observed range as a proxy: samples
    within `tolerance` (relative to that range) of either extreme are
    considered pinned. Best-effort heuristic, not hardware-calibrated.

    Dimensions carrying fewer than `min_distinct` distinct values are
    excluded, and the result is NaN if that leaves none. A constant field
    sits at both of its own extremes at once and a boolean one (a gripper
    open/close flag, a status bit) is at an extreme in *every* sample by
    construction, so both used to read as permanently 100% saturated -- and
    `health.clipping_frac` feeds the composite score, so a single status bit
    in a field group was enough to inflate it. Only a signal with room to
    move in between can meaningfully be said to have hit a rail. The trade
    is that a real saturation event captured at only two distinct levels is
    missed, accepted because the alternative flags every discrete field in
    every episode.

    Note also that this can't return 0: each scorable dimension's own argmin
    and argmax are pinned by definition, so a perfectly clean signal still
    reads ``2 / n_samples``.
    """
    if len(vectors) == 0:
        return np.nan

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        return np.nan

    distinct = np.array([len(np.unique(column[~np.isnan(column)])) for column in vectors.T])
    scorable = distinct >= min_distinct
    if not scorable.any():
        return np.nan

    # nanmin/nanmax, not min/max: health.clipping_frac fills fields a message
    # didn't carry with NaN, and one hole would otherwise poison the whole
    # column's range (and with it every comparison against it). Any column
    # that survived the distinct-value filter has real values to span.
    columns = vectors[:, scorable]
    lo, hi = np.nanmin(columns, axis=0), np.nanmax(columns, axis=0)
    span = hi - lo
    near_lo = np.abs(columns - lo) <= tolerance * span
    near_hi = np.abs(columns - hi) <= tolerance * span
    return float(np.mean(near_lo | near_hi))
