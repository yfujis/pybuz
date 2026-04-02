# pybuz/io.py
# authors: Yuki Fujishima
# last edited: 2025-10-20

# Description: Module for reading buzcode sleep state, ripple, and EMG data files.
# Matlab files that are version v7.3 or higher are read using h5py, while older versions are read using scipy.io.loadmat.
# Currently, the same buzcode generated files can be either format depending on the MATLAB version used to create them (or some other reasons).
from __future__ import annotations
import logging, h5py
from pathlib import Path
from numpy import ndarray
import numpy as np
from scipy.io import loadmat
from types import SimpleNamespace

from dataclasses import dataclass, asdict
import math
import glob
from typing import Iterable, List, Optional, Tuple, Union

import pynapple as nap

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SleepState:

    def __init__(self, basepath: Path):
        """recordingname.SleepState.states.mat
        & recordingname.SleepStateEpisodes.states.mat

        states is the "raw" state scoring. episodes are joined episodes of
        extended (40s) time in a given states, allowing for brief interruptions.
        also contains NREM packets, unitary epochs of NREM as described in Watson et al 2016
        Args:
            basepath (Path): path to the directory containing the recording
        """
        self.basepath = basepath
        try:
            mat_file = h5py.File(basepath / f"{basepath.stem}.SleepState.states.mat", 'r')
            self.WAKEstate: ndarray = mat_file['SleepState']['ints']['WAKEstate'][:].T
            self.NREMstate: ndarray = mat_file['SleepState']['ints']['NREMstate'][:].T
            self.REMstate: ndarray = mat_file['SleepState']['ints']['REMstate'][:].T
            try:
                self.MAstate: ndarray = mat_file['SleepState']['ints']['MAstate'][:].T
            except Exception as e:
                logger.warning(f"Could not find MAstate: {e}")
                self.MAstate = np.empty((0, 2))
            self.idx_states: ndarray = mat_file['SleepState']['idx']['states'][:].squeeze()
            self.idx_timestamps: ndarray = mat_file['SleepState']['idx']['timestamps'][:].squeeze()
            # self.detectorinfo = mat_file['SleepState']['detectorinfo']
            self.THchanID: int = int(mat_file['SleepState']['detectorinfo']['detectionparms']['SleepScoreMetrics']['THchanID'][:].flatten()[0])
            self.SWchanID: int = int(mat_file['SleepState']['detectorinfo']['detectionparms']['SleepScoreMetrics']['SWchanID'][:].flatten()[0])
            mat_file.close()
        except Exception as e:
            logger.error(f"Could not open {basepath / f'{basepath.stem}.SleepState.states.mat'}: {e}")
            logger.info("Trying to open with scipy.io.loadmat instead of h5py.File")
            mat_file = loadmat(basepath / f"{basepath.stem}.SleepState.states.mat")
            self.WAKEstate: ndarray = mat_file['SleepState']['ints'][0, 0]['WAKEstate'][0, 0]
            self.NREMstate: ndarray = mat_file['SleepState']['ints'][0, 0]['NREMstate'][0, 0]
            self.REMstate: ndarray = mat_file['SleepState']['ints'][0, 0]['REMstate'][0, 0]
            try:
                self.MAstate: ndarray = mat_file['SleepState']['ints'][0, 0]['MAstate'][0, 0]
            except Exception as e:
                logger.warning(f"Could not find MAstate: {e}")
                self.MAstate = np.empty((0, 2))
            self.idx_states: ndarray = mat_file['SleepState']['idx'][0, 0]['states'][0, 0].squeeze()
            self.idx_timestamps: ndarray = mat_file['SleepState']['idx'][0, 0]['timestamps'][0, 0].squeeze()
            self.THchanID: int = int(mat_file['SleepState']['detectorinfo'][0][0]['detectionparms'][0][0]['SleepScoreMetrics'][0][0]['THchanID'][0][0][0][0])
            self.SWchanID: int = int(mat_file['SleepState']['detectorinfo'][0][0]['detectionparms'][0][0]['SleepScoreMetrics'][0][0]['SWchanID'][0][0][0][0])

        self.statenames: dict = {1: 'WAKE', 3: 'NREM', 5: 'REM'}



        mat_episode = loadmat(basepath / f"{basepath.stem}.SleepStateEpisodes.states.mat")
        self.WAKEepisode: ndarray = mat_episode['SleepStateEpisodes']['ints'][0, 0]['WAKEepisode'][0, 0]
        self.REMepisode: ndarray = mat_episode['SleepStateEpisodes']['ints'][0, 0]['REMepisode'][0, 0]
        self.NREMepisode: ndarray = mat_episode['SleepStateEpisodes']['ints'][0, 0]['NREMepisode'][0, 0]
        self.MA: ndarray = mat_episode['SleepStateEpisodes']['ints'][0, 0]['MA'][0, 0]
        self.MA_REM: ndarray = mat_episode['SleepStateEpisodes']['ints'][0, 0]['MA_REM'][0, 0]
        self.NREMpacket: ndarray = mat_episode['SleepStateEpisodes']['ints'][0, 0]['NREMpacket'][0, 0]
        mat_episode = None

    def _convert_to_nap_intervalset(self, arr: ndarray) -> nap.IntervalSet:
        if arr.size == 0:
            return nap.IntervalSet([], [])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        return nap.IntervalSet(start=arr[:, 0], end=arr[:, 1])

    def convert_all_to_nap(self):
        self.WAKEstate = self._convert_to_nap_intervalset(self.WAKEstate)
        self.NREMstate = self._convert_to_nap_intervalset(self.NREMstate)
        self.REMstate = self._convert_to_nap_intervalset(self.REMstate)
        self.MAstate = self._convert_to_nap_intervalset(self.MAstate)
        self.WAKEepisode = self._convert_to_nap_intervalset(self.WAKEepisode)
        self.NREMepisode = self._convert_to_nap_intervalset(self.NREMepisode)
        self.REMepisode = self._convert_to_nap_intervalset(self.REMepisode)
        self.MA = self._convert_to_nap_intervalset(self.MA)
        self.MA_REM = self._convert_to_nap_intervalset(self.MA_REM)
        self.NREMpacket = self._convert_to_nap_intervalset(self.NREMpacket)

    def __repr__(self):
        return f"SleepState({self.basepath})"

def read_sleep_state(basepath: Path, return_as_pynapple: bool = True) -> SleepState:
    sleep_state = SleepState(basepath)
    if return_as_pynapple:
        sleep_state.convert_all_to_nap()
    return sleep_state


class Ripples:

    def __init__(self, basepath: Path):
        """recordingname.ripples.events.mat

        Args:
            basepath (Path): _description_
        """
        data = loadmat(basepath / f"{basepath.stem}.ripples.events.mat")['ripples']
        self.timestamps: ndarray = data['timestamps'][0, 0]
        self.peaks: ndarray = data['peaks'][0, 0]
        self.peak_normed_power: ndarray = data['peakNormedPower'][0, 0]
        self.stdev: float = data['stdev'][0, 0][0, 0]

        self.noise_times: ndarray = data['noise'][0, 0]['times'][0, 0]
        self.noise_peaks: ndarray = data['noise'][0, 0]['peaks'][0, 0]
        self.noise_peak_normed_power: ndarray = data['noise'][0, 0]['peakNormedPower'][0, 0]

    def __repr__(self):
        return f"Ripples({self.basepath})"

def read_ripples(basepath: Path) -> Ripples:
    return Ripples(basepath)


# class EMG:

#     def __init__(self, basepath: Path):
#         mat_file = loadmat(basepath / f"{basepath.stem}.EMGFromLFP.LFP.mat")
#         data = mat_file['EMGFromLFP']
#         self.timestamps: ndarray = data['timestamps'][0, 0].squeeze()
#         self.data: ndarray = data['data'][0, 0].squeeze()
#         self.detector_name: str = data['detectorName'][0, 0][0]
#         self.sampling_frequency: int = data['samplingFrequency'][0, 0][0][0]

#     def __repr__(self):
#         return f"EMG({self.basepath})"

# def read_emg(basepath: Path) -> EMG:
#     return EMG(basepath)


def read_lfp(basepath: Path, n_channels: int = None) -> np.ndarray:
    """
    Read .lfp file and return as a numpy array of shape (n_channels, n_samples)
    """
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.lfp"
    with open(file_path, 'rb') as fid:
        # Read the analog input data
        lfp = np.fromfile(fid, dtype=np.int16)  # now int16 not uint16
    if n_channels is None:
        sessionInfo = loadmat(basepath / f'{basepath.name}.sessionInfo.mat')
        n_channels = sessionInfo['sessionInfo'][0, 0]['nChannels'][0, 0]
    lfp = lfp.reshape(n_channels, -1, order='F')
    return lfp


# -------- Pretty tree printer --------
class _TreeReprMixin:
    def _tree_repr(self, obj=None, depth=0, max_depth=6, max_items=20, visited=None):
        if visited is None:
            visited = set()
        if obj is None:
            obj = self

        indent = "  " * depth
        nl = "\n"

        # Avoid infinite recursion on cycles
        oid = id(obj)
        if oid in visited:
            return f"{indent}<...cycle...>"
        visited.add(oid)

        # Base types
        if isinstance(obj, (str, int, float, bool, type(None), np.number, np.bool_)):
            return indent + repr(obj)

        # NumPy arrays: print summary
        if isinstance(obj, np.ndarray):
            desc = f"array(shape={obj.shape}, dtype={obj.dtype}"
            # small arrays preview
            if obj.size and obj.size <= 10 and obj.ndim <= 2:
                desc += f", preview={np.array2string(obj, threshold=10)})"
            else:
                desc += ")"
            return indent + desc

        # Dictionaries
        if isinstance(obj, dict):
            if depth >= max_depth:
                return indent + "{...}"
            lines = [indent + "{"]
            items = list(obj.items())
            clipped = len(items) > max_items
            for k, v in items[:max_items]:
                key_str = repr(k)
                val_str = self._tree_repr(v, depth+1, max_depth, max_items, visited)
                lines.append(f"{indent}  {key_str}:")
                lines.append(val_str)
            if clipped:
                lines.append(f"{indent}  ... ({len(items)-max_items} more)")
            lines.append(indent + "}")
            return nl.join(lines)

        # Lists / Tuples
        if isinstance(obj, (list, tuple)):
            if depth >= max_depth:
                return indent + ("[...]" if isinstance(obj, list) else "(...)")
            bracket_open = "[" if isinstance(obj, list) else "("
            bracket_close = "]" if isinstance(obj, list) else ")"
            lines = [indent + bracket_open]
            clipped = len(obj) > max_items
            for item in obj[:max_items]:
                lines.append(self._tree_repr(item, depth+1, max_depth, max_items, visited))
            if clipped:
                lines.append(f"{indent}  ... ({len(obj)-max_items} more)")
            lines.append(indent + bracket_close)
            return nl.join(lines)

        # PrettyNamespace or any object with __dict__
        if hasattr(obj, "__dict__"):
            if depth >= max_depth:
                return indent + f"{obj.__class__.__name__}(...)"
            lines = [indent + f"{obj.__class__.__name__}("]
            attrs = vars(obj)
            clipped = len(attrs) > max_items
            for name in sorted(list(attrs))[:max_items]:
                lines.append(f"{indent}  {name}:")
                lines.append(self._tree_repr(attrs[name], depth+1, max_depth, max_items, visited))
            if clipped:
                lines.append(f"{indent}  ... ({len(attrs)-max_items} more)")
            lines.append(indent + ")")
            return nl.join(lines)

        # Fallback
        return indent + repr(obj)


class PrettyNamespace(SimpleNamespace, _TreeReprMixin):
    def __repr__(self):
        return self._tree_repr(self)


# -------- MATLAB -> Python simplifier with PrettyNamespace --------
def _simplify(matobj):
    """Recursively simplify MATLAB structures, removing redundant brackets."""
    if isinstance(matobj, np.ndarray):
        # unwrap arrays of size 1
        if matobj.size == 1:
            # .item() can raise if dtype=object but empty; guard with try/except
            try:
                return _simplify(matobj.item())
            except Exception:
                return matobj
        else:
            # recursively simplify object arrays; keep numeric arrays as-is for efficiency
            if matobj.dtype == object:
                flat = [_simplify(x) for x in matobj.flat]
                return np.array(flat, dtype=object).reshape(matobj.shape)
            else:
                return matobj
    elif isinstance(matobj, np.void):  # MATLAB struct
        fields = {}
        for name in matobj.dtype.names or []:
            fields[name] = _simplify(matobj[name])
        return PrettyNamespace(**fields)
    else:
        return matobj

def _loadmat_any(filepath):
    """Load .mat files (v7.2 or v7.3) seamlessly."""
    try:
        # Try normal MATLAB file first (v7.2 or older)
        return loadmat(filepath, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        # MATLAB v7.3 HDF5-based file
        data = {}
        with h5py.File(filepath, "r") as f:
            def read_item(obj):
                # Convert MATLAB HDF5 structures recursively
                if isinstance(obj, h5py.Dataset):
                    arr = obj[()]
                    if arr.dtype.kind == 'S':  # bytes to str
                        arr = arr.astype(str)
                    return arr
                elif isinstance(obj, h5py.Group):
                    return {k: read_item(obj[k]) for k in obj.keys()}
                else:
                    return obj
            for key in f.keys():
                data[key] = read_item(f[key])
        return data

class MatObject(_TreeReprMixin):
    def __init__(self, filepath):
        # data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
        data = _loadmat_any(str(filepath))
        for key, val in data.items():
            if key.startswith('__'):  # skip system keys
                continue
            setattr(self, key, _simplify(val))

    # def __repr__(self):
    #     return self._tree_repr(self)
    def __repr__(self):
        keys = [k for k in self.__dict__.keys() if not k.startswith("_")]
        return f"MatObject({', '.join(keys)})"


# -------- Convenience loader --------
def read_session_info(basepath: Path) -> PrettyNamespace:
    """Read session info from <basename>.sessionInfo.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.sessionInfo.mat"
    return MatObject(file_path).sessionInfo


def read_ripple_events(basepath: Path) -> PrettyNamespace:
    """Read ripple events from <basename>.ripples.events.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.ripples.events.mat"
    return MatObject(file_path).ripples


def read_ripple_stats(basepath: Path) -> PrettyNamespace:
    """Read ripple stats from <basename>.ripples.stats.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.ripples.stats.mat"
    return MatObject(file_path).stats


def read_ripple_maps(basepath: Path) -> PrettyNamespace:
    """Read ripple maps from <basename>.ripples.maps.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.ripples.maps.mat"
    return MatObject(file_path).maps


def read_ripple_data(basepath: Path) -> PrettyNamespace:
    """Read ripple data from <basename>.ripples.data.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.ripples.data.mat"
    return MatObject(file_path).data

def read_emg(basepath: Path) -> PrettyNamespace:
    """Read EMG from LFP data from <basename>.EMGFromLFP.LFP.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.EMGFromLFP.LFP.mat"
    return MatObject(file_path).EMGFromLFP


def read_sleep_states(basepath: Path) -> PrettyNamespace:
    """ Read sleep states from <basename>.SleepState.states.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.SleepState.states.mat"
    return MatObject(file_path).SleepState


def read_spikes_cellinfo(basepath: Path) -> PrettyNamespace:
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.spikes.cellinfo.mat"
    return MatObject(file_path).spikes


def read_cellmetrics_cellinfo(basepath: Path) -> PrettyNamespace:
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.cell_metrics.cellinfo.mat"
    return MatObject(file_path).cell_metrics


@dataclass
class LFPInterval:
    data: np.ndarray              # shape: [Nt, Nd]
    timestamps: np.ndarray        # shape: [Nt]
    interval: Tuple[float, float] # (start, end) in seconds
    channels: np.ndarray          # shape: [Nd]
    sampling_rate: float          # Hz (after downsample)
    duration: float               # seconds
    filename: str                 # basename + extension actually loaded
    region: Optional[np.ndarray] = None  # optional, aligned with channels


class LFPFileNotFoundError(FileNotFoundError):
    pass


def _find_basefile(basepath: Path, basename: Optional[str], from_dat: bool) -> str:
    """
    Find the actual file to open (basename + extension).
    Preference:
      - if basename given: <basename>.lfp (or .dat) else fallback to .eeg
      - if basename not given: look for *lfp (or *dat) else fall back to *eeg
    Enforces a single match.
    """
    if basename:
        target_ext = "dat" if from_dat else "lfp"
        candidates = list((basepath / f"{basename}.{target_ext}",).copy())
        hits = [str(p) for p in candidates if Path(p).exists()]
        if not hits:
            # try .eeg
            eeg = basepath / f"{basename}.eeg"
            if eeg.exists():
                return eeg.name
            raise LFPFileNotFoundError(f"Could not find {basename}.lfp/.dat or {basename}.eeg in {basepath}")
        return Path(hits[0]).name
    else:
        # search by extension
        ext = "dat" if from_dat else "lfp"
        hits = glob.glob(str(basepath / f"*.{ext}"))
        if len(hits) == 0:
            # try .eeg
            hits = glob.glob(str(basepath / "*.eeg"))
            if len(hits) == 0:
                raise LFPFileNotFoundError(f"Could not find any *.{ext} or *.eeg file in {basepath}")
        if len(hits) > 1:
            raise RuntimeError(f"More than one *.{Path(hits[0]).suffix[1:]} file in {basepath}; please specify 'basename'.")
        return Path(hits[0]).name


def _get_sampling_rate(session_info, from_dat: bool) -> float:
    if from_dat:
        # wideband
        try:
            return float(session_info.rates.wideband)
        except AttributeError:
            return float(session_info["rates"]["wideband"])
    else:
        # lfp
        # try new name first, then legacy
        for keypath in (("lfpSampleRate",), ("rates", "lfp")):
            try:
                cur = session_info
                for k in keypath:
                    cur = getattr(cur, k) if hasattr(cur, k) else cur[k]
                return float(cur)
            except (KeyError, AttributeError, TypeError):
                continue
        raise KeyError("Could not determine LFP sampling rate from session_info (need lfpSampleRate or rates.lfp).")


def _get_nchannels(session_info) -> int:
    try:
        return int(session_info.nChannels)
    except AttributeError:
        return int(session_info["nChannels"])


def _get_all_channels(session_info) -> np.ndarray:
    try:
        ch = session_info.channels
    except AttributeError:
        ch = session_info["channels"]
    return np.asarray(ch, dtype=int)


def _get_regions(session_info, which_channels: np.ndarray) -> Optional[np.ndarray]:
    # Optional: return sessionInfo.region matched to order of which_channels (0-indexed)
    try:
        session_channels = _get_all_channels(session_info)
        session_regions = np.asarray(session_info.region)
        # stable intersection (preserve which_channels order)
        idx_map = {c: i for i, c in enumerate(session_channels)}
        region_idx = [idx_map[c] for c in which_channels if c in idx_map]
        if len(region_idx) != len(which_channels):
            return None
        return session_regions[region_idx]
    except Exception:
        return None


def _check_integer_downsample(sr: float, downsample: int):
    if downsample < 1 or (sr / downsample) % 1 != 0:
        raise ValueError("samplingRate/downsample must be an integer and downsample >= 1.")


def _file_total_samples(filepath: Path, dtype=np.int16, nchannels: int = 1) -> int:
    bytes_per_sample = np.dtype(dtype).itemsize
    sz = filepath.stat().st_size
    if sz % (bytes_per_sample * nchannels) != 0:
        raise ValueError(f"File size {sz} is not divisible by {bytes_per_sample}*{nchannels}.")
    return sz // (bytes_per_sample * nchannels)


def _load_binary_chunk(
    filepath: Path,
    start_samples: int,
    n_samples: int,
    nchannels: int,
    channels_0idx: Iterable[int],
    downsample: int = 1,
    dtype=np.int16,
) -> np.ndarray:
    """
    Reads a chunk from a multi-channel interleaved binary file, returns [Nt, Nd].
    Channels are 0-indexed IDs; file channels are in canonical [0..nchannels-1] order.
    """
    channels_0idx = np.asarray(list(channels_0idx), dtype=int)
    # memmap whole file (safe even for big files, but we only slice the needed rows)
    mm = np.memmap(filepath, dtype=dtype, mode='r')
    # reshape as [Ntotal_samples, nchannels]
    total = mm.size // nchannels
    arr = mm[: total * nchannels].reshape(total, nchannels)

    stop = start_samples + n_samples
    if stop > total:
        stop = total
    if start_samples >= stop:
        return np.empty((0, len(channels_0idx)), dtype=dtype)

    # downsample by simple decimation (like MATLAB 'downsample' with integer factor)
    sl = slice(start_samples, stop, downsample)
    out = arr[sl, :][:, channels_0idx].copy()  # copy to free the memmap
    del mm  # ensure file handle closed on some platforms
    return out


def get_lfp(
    channels: Union[str, Iterable[int]],            # 'all' or list-like of 0-indexed channel IDs
    *,
    basename: Optional[str] = None,
    intervals: Optional[Iterable[Iterable[float]]] = None,
    restrict: Optional[Iterable[Iterable[float]]] = None,
    basepath: Union[str, Path, None] = None,
    downsample: int = 1,
    no_prompts: bool = False,                       # kept for API parity
    from_dat: bool = False
) -> List[LFPInterval]:
    """
    Python translation of bz_GetLFP.

    Returns a list of LFPInterval (one per interval). If you passed a single interval,
    you'll still get a 1-element list (to match MATLAB behavior of lfp(i) over intervals).
    """
    basepath = Path(basepath) if basepath is not None else Path.cwd()

    # 1) Resolve file
    filename = _find_basefile(basepath, basename, from_dat)
    filepath = basepath / filename

    # 2) Session info & sampling rates
    session_info = read_session_info(basepath)
    nchannels = _get_nchannels(session_info)
    sr = _get_sampling_rate(session_info, from_dat=from_dat)
    _check_integer_downsample(sr, downsample)
    sr_out = sr / downsample

    # 3) Channels
    if isinstance(channels, str):
        if channels.lower() != "all":
            raise ValueError("channels must be a numeric iterable or 'all'.")
        channels_arr = _get_all_channels(session_info)
    else:
        channels_arr = np.asarray(list(channels), dtype=int)
        print(f"Loading Channels {channels_arr} (0-indexing, à la NeuroScope)")

    if channels_arr.min(initial=0) < 0 or channels_arr.max(initial=0) >= nchannels:
        raise IndexError(f"Requested channel outside [0, {nchannels-1}].")

    # 4) Intervals
    if intervals is None and restrict is None:
        intervals_arr = np.array([[0.0, math.inf]], dtype=float)
    elif intervals is None and restrict is not None:
        intervals_arr = np.asarray(restrict, dtype=float)
    else:
        intervals_arr = np.asarray(intervals, dtype=float)

    if intervals_arr.ndim == 1:
        intervals_arr = intervals_arr[None, :]  # shape -> (1, 2)

    # 5) Total samples for Inf handling
    total_samples = _file_total_samples(filepath, dtype=np.int16, nchannels=nchannels)

    # 6) Load each interval
    out: List[LFPInterval] = []
    for start_sec, end_sec in intervals_arr:
        # translate Inf to file end
        if math.isinf(end_sec):
            end_sec = total_samples / sr

        duration_sec = max(0.0, end_sec - start_sec)
        start_samples = int(round(start_sec * sr))
        n_samples = int(round(duration_sec * sr))

        data = _load_binary_chunk(
            filepath=filepath,
            start_samples=start_samples,
            n_samples=n_samples,
            nchannels=nchannels,
            channels_0idx=channels_arr,
            downsample=downsample,
            dtype=np.int16,
        )

        # timestamps correspond to the *downsampled* sampling rate
        if data.shape[0] > 0:
            n_out = data.shape[0]
            timestamps = (start_sec + np.arange(n_out) / sr_out).astype(float)
            interval_out = (start_sec, start_sec + (n_out - 1) / sr_out)
            duration_out = interval_out[1] - interval_out[0] if n_out > 1 else 0.0
        else:
            timestamps = np.array([], dtype=float)
            interval_out = (start_sec, start_sec)
            duration_out = 0.0

        region = _get_regions(session_info, channels_arr)

        out.append(
            LFPInterval(
                data=data,
                timestamps=timestamps,
                interval=interval_out,
                channels=channels_arr.copy(),
                sampling_rate=sr_out,
                duration=duration_out,
                filename=filename,
                region=region,
            )
        )

    return out


def read_lfp(
    basepath: Union[str, Path, None] = None,
    channels: Union[str, Iterable[int]] = None,            # 'all' or list-like of 0-indexed channel IDs
    restrict_ep: nap.IntervalSet = None,
    return_as_list: bool = False) -> nap.TsdFrame:
    """Read buzcode LFP data as a nap.TsdFrame.

    If channels is None, all channels are loaded. channels may be 'all' or an iterable
    of 0-indexed channel IDs. If restrict_ep is provided as a pynapple.IntervalSet its
    .values are used as the restrict intervals.
    """
    # normalize channels argument
    if channels is None:
        channels = "all"
    elif isinstance(channels, str):
        if channels.lower() != "all":
            raise ValueError("channels must be 'all' or an iterable of 0-indexed ints.")
    elif type(channels) is int:
        channels = [channels]
    else:
        try:
            # ensure we can iterate and materialize the sequence
            channels = list(channels)
        except TypeError:
            raise TypeError("channels must be 'all' or an iterable of 0-indexed ints.")
    # prepare basepath
    basepath = Path(basepath) if basepath is not None else Path.cwd()

    # Prepare restrict
    if restrict_ep is None:
        restrict = None
    elif type(restrict_ep) is nap.IntervalSet:
        restrict = restrict_ep.values
    else:
        restrict = np.array(restrict_ep, dtype=float)
        if restrict.ndim != 2 or restrict.shape[1] != 2:
            raise ValueError("restrict_ep must be an IntervalSet or a 2D array-like of shape (N, 2).")

    lfp_list = get_lfp(channels, restrict=restrict, basepath=basepath)
    lfp_tsdframe_list = [nap.TsdFrame(lfp_.timestamps, lfp_.data, columns=lfp_.channels) for lfp_ in lfp_list]
    if return_as_list:
        return lfp_tsdframe_list
    else:
        lfp_time_ = np.hstack([lfp_.timestamps for lfp_ in lfp_list])
        lfp_value_ = np.vstack([lfp_.data for lfp_ in lfp_list])
        w = np.hstack([True, ~np.array(np.diff(lfp_time_) == 0)])
        lfp  = nap.TsdFrame(lfp_time_[w], lfp_value_[w], columns=lfp_list[0].channels)
    return lfp


def read_chanCoords_channelinfo(basepath: Path) -> PrettyNamespace:
    """Read channel coordinates from <basename>.chanCoords.channelinfo.mat and return a PrettyNamespace."""
    basepath = Path(basepath)
    file_path = basepath / f"{basepath.stem}.chanCoords.channelinfo.mat"
    return MatObject(file_path).chanCoords