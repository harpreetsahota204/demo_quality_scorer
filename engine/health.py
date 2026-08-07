"""Sensor-health metrics: dropout, desync, clock drift, rate stability, clipping.

Computed directly from raw per-channel message timestamps (``log_time``,
``publish_time``), independent of any specific sensor type -- these apply
identically to a camera channel, a telemetry channel, or a JSON sidecar.
"""

import numpy as np

from .activity import pinned_fraction

# A gap more than this many multiples of the expected inter-arrival time
# counts as a dropout.
DROPOUT_GAP_MULTIPLE = 3.0


def rate_stats(records):
    """Returns ``(expected_dt_s, dropout_frac, rate_cov)`` from a channel's log-time gaps.

    Args:
        records: decoded records for one channel, as returned by
            :func:`.decode.decode_channel`

    Returns:
        a 3-tuple: the median inter-arrival time in seconds, the fraction of
        gaps exceeding ``DROPOUT_GAP_MULTIPLE`` times that, and the
        coefficient of variation of inter-arrival times (0 == perfectly
        steady)
    """
    if len(records) < 3:
        return np.nan, np.nan, np.nan

    # Diff while still int64 (exact); only convert to float seconds after the
    # subtraction, since raw nanosecond epoch timestamps (~1.7e18) exceed
    # float64's exact-integer range (2**53) and lose precision otherwise.
    log_times_ns = np.array([r[0] for r in records], dtype=np.int64)
    gaps = np.diff(log_times_ns).astype(np.float64) / 1e9
    expected_dt = float(np.median(gaps))
    if expected_dt <= 0:
        return expected_dt, np.nan, np.nan

    dropout_frac = float(np.mean(gaps > DROPOUT_GAP_MULTIPLE * expected_dt))
    rate_cov = float(np.std(gaps) / expected_dt)
    return expected_dt, dropout_frac, rate_cov


def clock_drift_ppm(records):
    """Linear trend of ``(log_time - publish_time)`` over the episode, in parts per million.

    Zero drift means the log/publish timebases stay in lockstep; a nonzero
    value indicates one clock runs fast/slow relative to the other.
    """
    if len(records) < 3:
        return np.nan

    # Subtract while still int64 (exact); huge nearly-equal nanosecond epoch
    # timestamps would otherwise lose precision if cast to float64 first.
    log_times_ns = np.array([r[0] for r in records], dtype=np.int64)
    publish_times_ns = np.array([r[1] for r in records], dtype=np.int64)
    if not np.any(publish_times_ns):
        return np.nan

    offsets_ns = (log_times_ns - publish_times_ns).astype(np.float64)
    elapsed_s = (log_times_ns - log_times_ns[0]).astype(np.float64) / 1e9
    if elapsed_s[-1] == 0:
        return np.nan

    slope_ns_per_s, _ = np.polyfit(elapsed_s, offsets_ns, 1)
    return float(slope_ns_per_s * 1e-3)  # (ns/s) -> fractional drift -> ppm


def desync_ms(records_a, records_b):
    """Median nearest-neighbor timestamp offset between two channels, in milliseconds.

    Approximates cross-channel sync by matching each message in the shorter
    channel to its nearest-in-time neighbor in the other, per the dataset
    authors' own recommendation to use nearest-neighbor timestamp matching
    rather than assume index alignment between independently-clocked
    channels.
    """
    if len(records_a) < 2 or len(records_b) < 2:
        return np.nan

    # Keep everything int64 through the nearest-neighbor search and only
    # convert to float after taking the (small) offset, for the same
    # precision reason as clock_drift_ppm above.
    a = np.array([r[0] for r in records_a], dtype=np.int64)
    b = np.array([r[0] for r in records_b], dtype=np.int64)
    if len(a) > len(b):
        a, b = b, a

    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    left, right = b[idx - 1], b[idx]
    nearest = np.where(np.abs(a - left) <= np.abs(a - right), left, right)
    offsets_ns = np.abs(a - nearest).astype(np.float64)
    return float(np.median(offsets_ns) / 1e6)


def clipping_frac(records, field_names):
    """Fraction of a channel's samples pinned at their observed extremes."""
    if not records:
        return np.nan
    vectors = np.array(
        [[fields.get(name, np.nan) for name in field_names] for _, _, fields in records]
    )
    return pinned_fraction(vectors)
