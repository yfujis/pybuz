from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, filtfilt, lfilter


@dataclass
class RippleDetectionParams:
    thresholds: Tuple[float, float] = (2.0, 5.0)
    durations: Tuple[float, float] = (30.0, 100.0)  # ms: (min inter-ripple interval, max ripple duration)
    min_duration: float = 20.0  # ms
    restrict: Optional[NDArray[np.float64]] = None  # Nx2 intervals in seconds
    frequency: float = 1250.0
    stdev: Optional[float] = None
    passband: Tuple[float, float] = (130.0, 200.0)
    emg_thresh: Optional[float] = 0.9


def _filter0(b: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replicate MATLAB Filter0 behavior used in bz_FindRipples."""
    if x.ndim == 1:
        x = x[:, None]

    if b.size % 2 != 1:
        raise ValueError("Filter order should be odd")

    shift = (b.size - 1) // 2
    y0, zf = lfilter(b, [1.0], x, axis=0, zi=np.zeros((b.size - 1, x.shape[1])))
    y = np.vstack((y0[shift:, :], zf[:shift, :]))
    return y[:, 0] if y.shape[1] == 1 else y


def _unity(a: NDArray[np.float64], sd: Optional[float], keep: Optional[NDArray[np.bool_]]) -> Tuple[NDArray[np.float64], float]:
    """Normalize to z-score, optionally using a restricted subset and/or forced stdev."""
    if keep is not None and keep.any():
        mean_a = float(np.mean(a[keep]))
        std_a = float(np.std(a[keep]))
    else:
        mean_a = float(np.mean(a))
        std_a = float(np.std(a))

    if sd is not None:
        std_a = float(sd)

    if std_a == 0:
        std_a = np.finfo(float).eps

    return (a - mean_a) / std_a, std_a


def _in_intervals(timestamps: NDArray[np.float64], intervals: NDArray[np.float64]) -> NDArray[np.bool_]:
    keep = np.zeros(timestamps.shape[0], dtype=bool)
    for start, stop in intervals:
        keep |= (timestamps >= start) & (timestamps <= stop)
    return keep


def _bandpass_filter(signal: NDArray[np.float64], fs: float, passband: Tuple[float, float], order: int = 3) -> NDArray[np.float64]:
    b, a = butter(order, passband, btype="bandpass", fs=fs)
    return filtfilt(b, a, signal)


def bz_find_ripples(
    lfp: ArrayLike,
    timestamps: ArrayLike,
    *,
    thresholds: Sequence[float] = (2.0, 5.0),
    durations: Sequence[float] = (30.0, 100.0),
    min_duration: float = 20.0,
    restrict: Optional[ArrayLike] = None,
    frequency: float = 1250.0,
    stdev: Optional[float] = None,
    noise: Optional[ArrayLike] = None,
    passband: Sequence[float] = (130.0, 200.0),
    emg: Optional[ArrayLike] = None,
    emg_timestamps: Optional[ArrayLike] = None,
    emg_thresh: Optional[float] = 0.9,
) -> Dict[str, Any]:
    """
    Python rewrite of buzcode's bz_FindRipples.m using array inputs.

    Parameters
    ----------
    lfp
        1D unfiltered LFP trace used for ripple detection.
    timestamps
        1D timestamps (seconds), same length as lfp.
    thresholds
        (low, high) thresholds in units of NSS stdev.
    durations
        (min inter-ripple interval [ms], max ripple duration [ms]).
    min_duration
        Min ripple duration (ms).
    restrict
        Optional Nx2 intervals (seconds) used to compute normalization stats.
    frequency
        Sampling rate in Hz.
    stdev
        Optional fixed stdev used for normalization.
    noise
        Optional noisy channel (1D signal) used for exclusion.
    passband
        Ripple-band passband in Hz.
    emg
        Optional EMG trace (typically scaled 0..1) used for exclusion.
    emg_timestamps
        EMG timestamps. If omitted and emg is provided, detection timestamps are reused.
    emg_thresh
        Exclude ripple if nearest EMG sample at ripple start is > emg_thresh.

    Returns
    -------
    dict
        Buzcode-like ripple event dictionary.
    """
    signal_raw = np.asarray(lfp, dtype=float).squeeze()
    ts = np.asarray(timestamps, dtype=float).squeeze()

    if signal_raw.ndim != 1 or ts.ndim != 1:
        raise ValueError("lfp and timestamps must be 1D")
    if signal_raw.shape[0] != ts.shape[0]:
        raise ValueError("lfp and timestamps must have the same length")
    if signal_raw.shape[0] < 3:
        raise ValueError("lfp is too short for ripple detection")

    low_threshold_factor = float(thresholds[0])
    high_threshold_factor = float(thresholds[1])
    min_inter_ripple_interval = float(durations[0])
    max_ripple_duration = float(durations[1])
    min_ripple_duration = float(min_duration)
    passband_tuple = (float(passband[0]), float(passband[1]))

    signal = _bandpass_filter(signal_raw, fs=frequency, passband=passband_tuple, order=3)

    # bz_FindRipples uses frequency/frequency*11, which is always 11 samples.
    window_length = 11
    window = np.ones(window_length, dtype=float) / window_length

    squared_signal = signal**2

    keep = None
    if restrict is not None:
        restrict_arr = np.asarray(restrict, dtype=float)
        if restrict_arr.ndim != 2 or restrict_arr.shape[1] != 2:
            raise ValueError("restrict must be shaped (N,2)")
        keep = _in_intervals(ts, restrict_arr)

    nss_input = _filter0(window, squared_signal)
    normalized_squared_signal, sd_used = _unity(nss_input, stdev, keep)

    # First pass: threshold crossing detection
    thresholded = normalized_squared_signal > low_threshold_factor
    d = np.diff(thresholded.astype(np.int8))
    start = np.where(d > 0)[0] + 1
    stop = np.where(d < 0)[0]

    if stop.size == start.size - 1:
        start = start[:-1]
    if stop.size - 1 == start.size:
        stop = stop[1:]
    if start.size and stop.size and start[0] > stop[0]:
        stop = stop[1:]
        start = start[:-1]

    if start.size == 0 or stop.size == 0:
        return _empty_output(sd_used, thresholds, durations, min_duration, restrict, frequency, passband_tuple)

    first_pass = np.column_stack((start, stop))

    # Second pass: merge close events
    min_inter_ripple_samples = int(round((min_inter_ripple_interval / 1000.0) * frequency))
    second_pass = []
    ripple = first_pass[0].copy()
    for i in range(1, first_pass.shape[0]):
        if first_pass[i, 0] - ripple[1] < min_inter_ripple_samples:
            ripple[1] = first_pass[i, 1]
        else:
            second_pass.append(ripple.copy())
            ripple = first_pass[i].copy()
    second_pass.append(ripple.copy())
    second_pass_arr = np.asarray(second_pass, dtype=int)

    # Third pass: peak threshold
    third_pass = []
    peak_normalized_power = []
    for s_idx, e_idx in second_pass_arr:
        segment = normalized_squared_signal[s_idx : e_idx + 1]
        max_value = float(np.max(segment))
        if max_value > high_threshold_factor:
            third_pass.append((s_idx, e_idx))
            peak_normalized_power.append(max_value)

    if len(third_pass) == 0:
        return _empty_output(sd_used, thresholds, durations, min_duration, restrict, frequency, passband_tuple)

    third_pass_arr = np.asarray(third_pass, dtype=int)
    peak_normalized_power_arr = np.asarray(peak_normalized_power, dtype=float)

    # Peak position: negative peak in filtered signal
    peak_position = np.empty(third_pass_arr.shape[0], dtype=int)
    for i, (s_idx, e_idx) in enumerate(third_pass_arr):
        seg = signal[s_idx : e_idx + 1]
        peak_position[i] = s_idx + int(np.argmin(seg))

    ripples = np.column_stack(
        (
            ts[third_pass_arr[:, 0]],
            ts[peak_position],
            ts[third_pass_arr[:, 1]],
            peak_normalized_power_arr,
        )
    )

    duration_sec = ripples[:, 2] - ripples[:, 0]
    keep_duration = (duration_sec <= max_ripple_duration / 1000.0) & (duration_sec >= min_ripple_duration / 1000.0)
    ripples = ripples[keep_duration]

    bad_rows = np.empty((0, 4), dtype=float)

    # Noise-based exclusion
    if noise is not None and ripples.shape[0] > 0:
        noise_signal_raw = np.asarray(noise, dtype=float).squeeze()
        if noise_signal_raw.shape[0] != ts.shape[0]:
            raise ValueError("noise must have the same length as lfp")

        squared_noise = _bandpass_filter(noise_signal_raw, fs=frequency, passband=passband_tuple, order=3) ** 2
        normalized_squared_noise, _ = _unity(_filter0(window, squared_noise), sd_used, None)

        excluded = np.zeros(ripples.shape[0], dtype=bool)
        for i in range(ripples.shape[0]):
            s_time, _, e_time, _ = ripples[i]
            lo = int(np.searchsorted(ts, s_time, side="left"))
            hi = int(np.searchsorted(ts, e_time, side="right"))
            if hi > lo and np.any(normalized_squared_noise[lo:hi] > high_threshold_factor):
                excluded[i] = True

        if excluded.any():
            bad_rows = ripples[excluded]
            ripples = ripples[~excluded]

    # EMG-based exclusion
    if emg is not None and emg_thresh is not None and ripples.shape[0] > 0:
        emg_values = np.asarray(emg, dtype=float).squeeze()
        if emg_timestamps is None:
            emg_ts = ts
        else:
            emg_ts = np.asarray(emg_timestamps, dtype=float).squeeze()

        if emg_values.shape[0] != emg_ts.shape[0]:
            raise ValueError("emg and emg_timestamps must have the same length")

        excluded = np.zeros(ripples.shape[0], dtype=bool)
        for i in range(ripples.shape[0]):
            ripple_start = ripples[i, 0]
            emg_idx = int(np.argmin(np.abs(emg_ts - ripple_start)))
            if emg_values[emg_idx] > float(emg_thresh):
                excluded[i] = True

        if excluded.any():
            bad_rows = np.vstack((bad_rows, ripples[excluded])) if bad_rows.size else ripples[excluded]
            bad_rows = bad_rows[np.argsort(bad_rows[:, 0])]
            ripples = ripples[~excluded]

    detector_params: Dict[str, Any] = {
        "thresholds": tuple(float(v) for v in thresholds),
        "durations": tuple(float(v) for v in durations),
        "min_duration": float(min_duration),
        "restrict": None if restrict is None else np.asarray(restrict, dtype=float),
        "frequency": float(frequency),
        "stdev": None if stdev is None else float(stdev),
        "passband": passband_tuple,
        "emg_thresh": None if emg_thresh is None else float(emg_thresh),
    }

    out = {
        "timestamps": ripples[:, [0, 2]] if ripples.size else np.empty((0, 2), dtype=float),
        "peaks": ripples[:, 1] if ripples.size else np.empty((0,), dtype=float),
        "peakNormedPower": ripples[:, 3] if ripples.size else np.empty((0,), dtype=float),
        "stdev": float(sd_used),
        "noise": {
            "times": bad_rows[:, [0, 2]] if bad_rows.size else np.empty((0, 2), dtype=float),
            "peaks": bad_rows[:, 1] if bad_rows.size else np.empty((0,), dtype=float),
            "peakNormedPower": bad_rows[:, 3] if bad_rows.size else np.empty((0,), dtype=float),
        },
        "detectorinfo": {
            "detectorname": "bz_FindRipples",
            "detectiondate": date.today().isoformat(),
            "detectionintervals": None if restrict is None else np.asarray(restrict, dtype=float),
            "detectionparms": detector_params,
            "detectionchannel": np.nan,
            "noisechannel": np.nan if noise is None else "array",
        },
    }

    return out


def _empty_output(
    sd_used: float,
    thresholds: Sequence[float],
    durations: Sequence[float],
    min_duration: float,
    restrict: Optional[ArrayLike],
    frequency: float,
    passband_tuple: Tuple[float, float],
) -> Dict[str, Any]:
    detector_params: Dict[str, Any] = {
        "thresholds": tuple(float(v) for v in thresholds),
        "durations": tuple(float(v) for v in durations),
        "min_duration": float(min_duration),
        "restrict": None if restrict is None else np.asarray(restrict, dtype=float),
        "frequency": float(frequency),
        "stdev": float(sd_used),
        "passband": passband_tuple,
    }

    return {
        "timestamps": np.empty((0, 2), dtype=float),
        "peaks": np.empty((0,), dtype=float),
        "peakNormedPower": np.empty((0,), dtype=float),
        "stdev": float(sd_used),
        "noise": {
            "times": np.empty((0, 2), dtype=float),
            "peaks": np.empty((0,), dtype=float),
            "peakNormedPower": np.empty((0,), dtype=float),
        },
        "detectorinfo": {
            "detectorname": "bz_FindRipples",
            "detectiondate": date.today().isoformat(),
            "detectionintervals": None if restrict is None else np.asarray(restrict, dtype=float),
            "detectionparms": detector_params,
            "detectionchannel": np.nan,
            "noisechannel": np.nan,
        },
    }
