"""Sensor-health metrics: dropout, desync, clock drift, rate stability, clipping.

Computed directly from raw per-channel message timestamps (``log_time``,
``publish_time``), independent of any specific sensor type -- these apply
identically to a camera channel, a telemetry channel, or a JSON sidecar.
"""

import numpy as np

from .activity import pinned_fraction
from .normalize import MAD_TO_STD

# A gap more than this many multiples of the expected inter-arrival time
# counts as a dropout.
DROPOUT_GAP_MULTIPLE = 3.0

# Percentile of nearest-neighbor timestamp offsets reported as desync. Low,
# deliberately -- see desync_ms.
DESYNC_PERCENTILE = 10


def rate_stats(records):
    """Returns ``(expected_dt_s, dropout_frac, rate_cov)`` from a channel's log-time gaps.

    Args:
        records: decoded records for one channel, as returned by
            :func:`.decode.decode_channel`

    Returns:
        a 3-tuple: the median inter-arrival time in seconds, the estimated
        fraction of expected messages that never arrived, and a robust
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

    # Weighted by how many messages each long gap swallowed, not merely by how
    # many gaps were long: a 100-message dropout and a 4-message one are very
    # different events that score identically if you only count gaps.
    long_gaps = gaps[gaps > DROPOUT_GAP_MULTIPLE * expected_dt]
    n_missed = float(np.sum(long_gaps / expected_dt - 1.0))
    dropout_frac = n_missed / (len(records) + n_missed)

    # MAD-based rather than std/mean: one dropout gap dominates a plain
    # standard deviation, and that same gap is already counted by
    # dropout_frac, so the two metrics would double-count a single event in
    # the composite score.
    jitter = float(np.median(np.abs(gaps - expected_dt)))
    rate_cov = MAD_TO_STD * jitter / expected_dt
    return expected_dt, float(dropout_frac), float(rate_cov)


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
    """Best-case nearest-neighbor timestamp offset between two channels, in milliseconds.

    Approximates cross-channel sync by matching each message in the sparser
    channel to its nearest-in-time neighbor in the other, per the dataset
    authors' own recommendation to use nearest-neighbor timestamp matching
    rather than assume index alignment between independently-clocked
    channels.

    Reports the :data:`DESYNC_PERCENTILE`th percentile of those offsets
    rather than the median, because the median carries a sampling floor it
    can't see past: two perfectly synchronized channels at 30 Hz and 10 Hz
    still show a median offset around a quarter of the denser channel's
    period (~8 ms), purely because the sparse channel's timestamps land at
    arbitrary phase between the dense one's. That made the metric mostly a
    reading of rate mismatch. A low percentile is instead the best alignment
    actually achieved anywhere in the episode: ~0 for a synchronized pair at
    any two rates, and still ~S for a pair genuinely skewed by S. The trade
    is a blind spot for *intermittent* desync -- a channel aligned for a
    tenth of the episode and adrift for the rest reads as aligned --
    accepted because the alternative flags every mixed-rate dataset.

    Nearest-neighbor matching itself can only resolve skew modulo the denser
    channel's sample interval: a channel shifted by exactly one or two of the
    dense channel's periods lands on its timestamps again and reads as
    perfectly aligned. That limitation is inherent to timestamp matching
    without a shared reference event and applies at any percentile.
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
    return float(np.percentile(offsets_ns, DESYNC_PERCENTILE) / 1e6)


def clipping_frac(records, field_names):
    """Fraction of a channel's samples pinned at their observed extremes.

    The health family's view of the same measurement
    `activity.pinned_fraction` provides -- over a whole channel's message
    stream rather than one motion window -- so a saturating sensor is caught
    even on channels nobody selected for motion scoring. Extremes are the
    values actually observed, not a datasheet range, since no sensor limits
    are available from an MCAP file; the exclusions that keep that heuristic
    honest live in `pinned_fraction`.
    """
    if not records:
        return np.nan
    vectors = np.array(
        [[fields.get(name, np.nan) for name in field_names] for _, _, fields in records]
    )
    return pinned_fraction(vectors)
