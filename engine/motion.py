"""Motion smoothness metrics: SPARC, LDLJ, RMS jerk, PSD low/high band ratio.

All four operate on a 1-D speed profile derived from a window's raw
field-group vectors (see :func:`speed_profile`), so they apply to any
numeric trajectory -- joint positions, end-effector pose, or anything else
discovery/decode surfaced -- not just a specific sensor's fields.

Source: Balasubramanian et al., "A robust and sensitive metric for
quantifying movement smoothness" (IEEE TBME 2012) for SPARC/LDLJ; PSD via
``scipy.signal.welch``.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch

# Metric polarity: whether a *higher* raw value means worse quality. Used by
# normalize.py to align every metric's z-score so "higher z == worse".
HIGHER_IS_WORSE = {
    "sparc": False,  # less negative (closer to 0) is smoother
    "ldlj": False,  # less negative is smoother
    "jerk_rms": True,
    "psd_lf_hf": False,  # a higher log low/high ratio means less high-frequency energy
}

# Shared zero-phase low-pass filter order, used by both jerk_rms (to filter
# before differentiating) and engine.activity.idle_frac (to smooth noise
# near zero-crossings). Public (not module-private) since both modules use
# it and the trim math in jerk_rms depends on the same value.
FILTER_ORDER = 4


def lowpass_filtfilt(signal, cutoff_hz, fs, order=FILTER_ORDER):
    """Zero-phase Butterworth low-pass filter, skipped gracefully when ill-defined.

    Returns `signal` unchanged (no filtering) when `cutoff_hz` isn't below
    the Nyquist frequency or `signal` is too short for filtfilt's padding
    requirement, rather than raising -- callers already have their own
    minimum-length guards for the metric computation itself.
    """
    nyquist = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return signal

    b, a = butter(order, cutoff_hz, fs=fs)
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal

    return filtfilt(b, a, signal)


def speed_profile(vectors, group, fs):
    """Derives a 1-D speed profile from a window's ``(n_samples, n_dims)`` vectors.

    If the field-group's name already suggests a rate (contains "vel" or
    "speed"), its vector norm is used directly. Otherwise the vectors are
    treated as a position-like trajectory and differentiated, since
    SPARC/LDLJ are defined over a velocity/speed profile, not a raw position
    signal.

    The difference is divided by the sample interval (multiplied by `fs`), so
    a position-derived profile comes out in units per *second* the way a
    rate-named group's already does, rather than units per sample. SPARC,
    LDLJ and psd_lf_hf are all invariant to that constant factor, but
    jerk_rms isn't: without it the value reported for a position channel is
    off by a factor of `fs` and isn't physical jerk.

    Args:
        vectors: an ``(n_samples, n_dims)`` array
        group: the field-group name the vectors came from
        fs: the channel's sampling rate, in Hz

    Returns:
        a 1-D array, one shorter than ``vectors`` if differentiated
    """
    if "vel" in group.lower() or "speed" in group.lower():
        return np.linalg.norm(vectors, axis=1)
    return np.linalg.norm(np.diff(vectors, axis=0), axis=1) * fs


def sparc(speed, fs, fc=10.0, amp_threshold=0.05, pad_level=4):
    """Spectral arc length of a speed profile (higher/less-negative = smoother).

    Measures how convoluted the movement's magnitude spectrum is: smooth
    motion is a few dominant low frequencies (a short arc), while corrections,
    tremor and stop-start motion add ripple that lengthens the curve. Negated
    so it reads as a score.

    Deliberately duration- and amplitude-invariant, which is what makes it the
    most trustworthy of the four metrics here: the spectrum is normalized to
    its own peak (amplitude drops out) and the frequency axis is normalized to
    the measured band's own width (duration drops out). Unlike `ldlj`, its
    values are comparable across windows and datasets.

    Args:
        speed: a 1-D speed profile
        fs: sampling rate, in Hz
        fc (10.0): band limit, in Hz. Sets what counts as movement rather
            than noise -- human voluntary motion lives below ~6 Hz, so 10 Hz
            suits teleop; raise it for high-bandwidth platforms. Values
            computed at different `fc` are not comparable
        amp_threshold (0.05): spectral magnitude below which content is
            treated as noise floor rather than movement
        pad_level (4): zero-padding exponent, for arc-length resolution
    """
    if len(speed) < 4 or fs <= 0:
        return np.nan

    # Heavy zero-padding interpolates the spectrum. Arc length is a
    # path-length measure, so it is systematically underestimated when
    # sampled coarsely -- padding makes the estimate stable across window
    # lengths instead of varying with the FFT's native bin spacing.
    n = len(speed)
    nfft = int(2 ** (np.ceil(np.log2(n)) + pad_level))
    spectrum = np.abs(np.fft.fft(speed, nfft))[: nfft // 2]
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)[: nfft // 2]

    peak = spectrum.max()
    if peak == 0:
        return np.nan
    spectrum = spectrum / peak

    # Trim to the band that actually carries movement. Including the noise
    # floor out to fc would add arc length proportional to how quiet the
    # channel is, making a still episode score as the roughest one.
    in_band = freqs <= fc
    above_threshold = in_band & (spectrum >= amp_threshold)
    if not above_threshold.any():
        return np.nan

    lo, hi = np.argmax(above_threshold), len(above_threshold) - 1 - np.argmax(above_threshold[::-1])
    band_freqs, band_spectrum = freqs[lo : hi + 1], spectrum[lo : hi + 1]
    if len(band_freqs) < 2:
        return np.nan

    freq_span = band_freqs[-1] - band_freqs[0]
    if freq_span == 0:
        return np.nan

    d_freq = np.diff(band_freqs) / freq_span
    d_amp = np.diff(band_spectrum)
    return float(-np.sum(np.sqrt(d_freq**2 + d_amp**2)))


def ldlj(speed, fs):
    """Log dimensionless jerk of a speed profile (higher/less-negative = smoother).

    Note the strong duration dependence: dimensionless jerk carries a
    ``duration**3`` factor and its integral grows with duration too, so for a
    stationary signal the whole quantity scales as roughly ``duration**4``.
    Comparing LDLJ across unequal spans is meaningless -- halving the span
    adds about 2.8 to the result on identical motion. This is why
    :func:`.windowing.windows_for_group` emits only complete windows.
    """
    if len(speed) < 5 or fs <= 0:
        return np.nan

    peak = np.max(np.abs(speed))
    if peak == 0:
        return np.nan

    dt = 1.0 / fs
    duration = (len(speed) - 1) * dt  # the span between first and last sample
    jerk = np.diff(speed, n=2) / dt**2
    dimensionless_jerk = (duration**3 / peak**2) * np.sum(jerk**2) * dt
    if dimensionless_jerk <= 0:
        return np.nan
    return float(-np.log(dimensionless_jerk))


def jerk_rms(speed, fs, cutoff_hz=10.0):
    """RMS jerk of a zero-phase-filtered speed profile (lower = smoother).

    Differentiating a speed profile twice (to get jerk) amplifies noise by
    omega^2, so an unfiltered RMS jerk on a real sensor signal is mostly
    noise, not signal. Low-pass filters first (`cutoff_hz` defaults to
    SPARC's own `fc`), then trims the samples most affected by filtfilt's
    edge padding before differentiating.
    """
    if len(speed) < 5 or fs <= 0:
        return np.nan

    filtered = lowpass_filtfilt(speed, cutoff_hz, fs)
    trim = min(len(filtered) // 4, 3 * FILTER_ORDER)
    trimmed = filtered[trim : len(filtered) - trim] if trim > 0 else filtered
    if len(trimmed) < 5:
        return np.nan

    jerk = np.diff(trimmed, n=2) * fs**2
    return float(np.sqrt(np.mean(jerk**2)))


def psd_lf_hf(speed, fs, cutoff_hz=None):
    """Log ratio of low-band to high-band power spectral density (lower = rougher).

    This is our own metric (a Welch low/high band-power ratio on the speed
    profile), not a reproduction of Sojib & Begum's PSD data-quality metric
    (arXiv:2605.01544), whose "PSD" is raw summed DFT power on 3D
    end-effector *position* (not a Welch density, not a ratio), ranked
    ascending. The name overlap is coincidental; the numbers are not
    comparable to that paper's. Ours trades away the paper's exact
    reproduction for being more variance-robust in production.

    Returned as a natural log, which makes it symmetric and unbounded in both
    directions. The bare ratio is bounded on its *bad* side -- it can only
    fall from the corpus median to 0, while a smooth window's ratio runs to
    arbitrarily large values -- so a robust z-score fit on it could never
    reach the warn threshold in the rough direction no matter how rough the
    motion got. Measured on real per-window data (Voxel51/ABC-130k) the bare
    ratio flagged 0.00% of windows on every channel: it was structurally
    incapable of firing.
    """
    if len(speed) < 8 or fs <= 0:
        return np.nan

    freqs, power = welch(speed, fs=fs, nperseg=min(len(speed), 64))
    cutoff_hz = cutoff_hz if cutoff_hz is not None else fs / 4.0
    low = power[freqs <= cutoff_hz].sum()
    high = power[freqs > cutoff_hz].sum()
    if low <= 0 or high <= 0:
        return np.nan
    return float(np.log(low / high))
