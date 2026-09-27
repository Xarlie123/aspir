"""SPIHIM adapter: raw acquisitions + Hadamard PNG bank -> :class:`MeasuredData`.

SPIHIM is the open experimental single-pixel collection of the OpenSpyrit
ecosystem: Beneti Martins et al., "OpenSpyrit: an ecosystem for open
single-pixel hyperspectral imaging", Opt. Express 31(10), 15599 (2023),
DOI 10.1364/OE.483937, licence CC-BY 4.0. Download it with
``scripts/download_spihim.py``.

Each ``*_raw.mat`` holds one spectrum per projected DMD pattern (``M``,
``(n_patterns, 2048)`` uint16) and the 1-based index of each pattern in the
8192-PNG bank ``Hadamard_64x64`` (``par.pattern_indices``, in acquisition
order). PNG ``2k+1`` / ``2k+2`` (1-based) are the positive / negative halves of
bipolar Hadamard row ``k``; their difference is +-1 and the dark offset of the
spectrometer cancels in the difference of the two measurements.
"""

from pathlib import Path

import numpy as np

from simulation_engine._1_dataset_gen.measured import MeasuredData

SPIHIM_DOI = "10.1364/OE.483937"
SPIHIM_CITATION = ("Beneti Martins et al., OpenSpyrit: an ecosystem for open single-pixel "
                   "hyperspectral imaging, Opt. Express 31(10), 15599 (2023)")
SPIHIM_LICENCE = "CC-BY 4.0"
INSTRUMENT = ("DMD Vialux V-7001; BWTek Exemplar spectrometer (2048 bins) via fibre; "
              "white LED + ND filter wheel; telecentric lens")

# uint16 ceiling of the spectrometer: a bin at this value is clipped.
SATURATION = 65535

# Orientation that makes the reconstruction match the physical scene (the
# printed text of the Thorlabs R1L1S2P star target reads correctly). The
# patterns are rotated by ``rot90`` quarter turns (numpy.rot90 on the image
# axes) and then mirrored left-right if ``flip``. Of the eight candidates only
# the four unflipped ones read the text un-mirrored, and two quarter turns
# bring it closest to upright (see docs/user_guide/experimental_data.md).
DEFAULT_ORIENTATION = {"rot90": 2, "flip": False}


class HadamardPngBank:
    """The 8192-PNG pattern bank, as 4096 bipolar +-1 rows.

    The bipolar stack is cached next to the PNGs (``bipolar_rows.npz``) after
    the first read, which takes a few seconds.
    """

    def __init__(self, root):
        root = Path(root)
        pngs = root / "Hadamard_64x64"
        self.png_dir = pngs if pngs.is_dir() else root
        self.cache = self.png_dir / "bipolar_rows.npz"
        self._bipolar = None

    def bipolar(self) -> np.ndarray:
        if self._bipolar is None:
            if self.cache.exists():
                with np.load(self.cache) as npz:
                    self._bipolar = npz["rows"]
            else:
                self._bipolar = self._read_pngs()
                np.savez_compressed(self.cache, rows=self._bipolar)
        return self._bipolar

    def _read_pngs(self) -> np.ndarray:
        from PIL import Image
        first = np.asarray(Image.open(self.png_dir / "Hadamard_64x64_1.png"))
        n = len(list(self.png_dir.glob("Hadamard_64x64_*.png")))
        if n % 2:
            raise ValueError(f"expected an even number of PNGs in {self.png_dir}, found {n}")
        stack = np.empty((n, *first.shape), dtype=np.int8)
        for i in range(n):
            img = np.asarray(Image.open(self.png_dir / f"Hadamard_64x64_{i + 1}.png"))
            stack[i] = (img > 0).astype(np.int8)       # PNGs are 0 / 255
        rows = stack[0::2] - stack[1::2]
        if not np.all(np.abs(rows) == 1):
            raise ValueError("positive / negative PNGs are not complementary")
        return rows


def read_raw_mat(path) -> dict:
    """The fields of a SPIHIM ``*_raw.mat`` as plain Python / NumPy values."""
    import scipy.io as sio
    d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    out = {"M": np.asarray(d["M"]), "lambda_vec": np.asarray(d["spec"].lambda_vec, dtype=np.float64)}
    for group in ("opt", "par"):
        obj = d[group]
        out[group] = {name: getattr(obj, name) for name in obj._fieldnames}
    out["pattern_indices"] = np.asarray(out["par"]["pattern_indices"], dtype=np.int64)
    return out


def saturated_bins(raw) -> np.ndarray:
    """Spectral bins clipped at the uint16 ceiling in any pattern."""
    return np.nonzero((raw["M"] == SATURATION).any(axis=0))[0]


def orient_patterns(patterns, rot90=0, flip=False) -> np.ndarray:
    """Rotate / mirror each pattern; the result is still a stack of +-Hadamard rows."""
    out = np.rot90(patterns, k=int(rot90), axes=(1, 2))
    if flip:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _bin_mask(lambda_vec, spectral, band, bin_index):
    if spectral == "sum":
        return np.ones_like(lambda_vec, dtype=bool)
    if spectral == "band":
        if band is None:
            raise ValueError("spectral='band' needs band=(lo_nm, hi_nm)")
        return (lambda_vec >= band[0]) & (lambda_vec <= band[1])
    if spectral == "bin":
        if bin_index is None:
            raise ValueError("spectral='bin' needs bin_index")
        mask = np.zeros_like(lambda_vec, dtype=bool)
        mask[int(bin_index)] = True
        return mask
    raise ValueError(f"spectral must be 'sum', 'band' or 'bin', got {spectral!r}")


def load_spihim(mat_path, bank, spectral="sum", band=None, bin_index=None,
                orientation=None, exclude_saturated=True, exclude_bins=None,
                flat_field_mat=None, reference=None) -> MeasuredData:
    """Convert one SPIHIM acquisition into :class:`MeasuredData`.

    Parameters
    ----------
    mat_path : path of the ``*_raw.mat`` acquisition.
    bank : :class:`HadamardPngBank` (or its directory).
    spectral : ``"sum"`` (default, panchromatic), ``"band"`` (``band=(lo, hi)``
        in nm) or ``"bin"`` (``bin_index``).
    orientation : ``{"rot90": k, "flip": bool}``; defaults to
        :data:`DEFAULT_ORIENTATION`.
    exclude_saturated : drop bins clipped at the uint16 ceiling in any pattern
        of this acquisition (a clipped bin no longer measures the light
        linearly). ``exclude_bins`` adds bins to drop, e.g. the union over a
        series of repeats so every repeat uses the same bins.
    flat_field_mat : optional ``noObject*`` acquisition of the same session and
        optical density; its full linear reconstruction is stored as
        ``flat_field``.
    reference : optional ``(H, W)`` reference image to carry along.

    Returns one frame: ``y[k]`` is the reduced spectrum of the positive half of
    row ``k`` minus that of its negative half, and the rows are in acquisition
    order.
    """
    if not isinstance(bank, HadamardPngBank):
        bank = HadamardPngBank(bank)
    orientation = dict(DEFAULT_ORIENTATION if orientation is None else orientation)
    raw = read_raw_mat(mat_path)

    bins = _bin_mask(raw["lambda_vec"], spectral, band, bin_index)
    dropped = set()
    if exclude_saturated:
        dropped |= set(saturated_bins(raw).tolist())
    if exclude_bins is not None:
        dropped |= set(np.asarray(exclude_bins).tolist())
    if dropped:
        bins[list(dropped)] = False
    if not bins.any():
        raise ValueError("no spectral bin left after the selection")

    y, rows = _differential(raw, bins)
    patterns = orient_patterns(bank.bipolar()[rows], **orientation)

    flat = None
    if flat_field_mat is not None:
        flat_raw = read_raw_mat(flat_field_mat)
        fy, frows = _differential(flat_raw, bins)
        fp = orient_patterns(bank.bipolar()[frows], **orientation)
        flat = np.tensordot(fy, fp.astype(np.float64), axes=(0, 0)) / len(fy)

    par, opt, lv = raw["par"], raw["opt"], raw["lambda_vec"]
    metadata = {
        "source": "SPIHIM", "source_file": Path(mat_path).name,
        "source_doi": SPIHIM_DOI, "citation": SPIHIM_CITATION, "licence": SPIHIM_LICENCE,
        "instrument": INSTRUMENT,
        "integration_time_s": float(par.get("CT", np.nan)),
        "dmd_frame_rate_hz": float(par.get("DMD_fr", np.nan)),
        "acquisition_type": str(opt.get("type", "")),
        "compression_ratio": float(opt.get("cr", np.nan)),
        "units": "spectrometer counts summed over the selected bins, positive minus negative",
        "spectral": spectral, "band_nm": list(band) if band is not None else None,
        "bin_index": bin_index,
        "bins_used": int(bins.sum()),
        "bins_dropped_saturated": len(dropped),
        "wavelength_range_used_nm": [float(lv[bins].min()), float(lv[bins].max())],
        "orientation": orientation,
        "flat_field_source": Path(flat_field_mat).name if flat_field_mat is not None else None,
    }
    return MeasuredData(y=y, patterns=patterns, img_size=patterns.shape[1],
                        acquisition_order=np.arange(len(rows)), reference=reference,
                        flat_field=flat, metadata=metadata)


def _differential(raw, bins):
    """Positive-minus-negative bucket values per Hadamard row, in acquisition order."""
    s = raw["M"][:, bins].astype(np.float64).sum(axis=1)
    p = raw["pattern_indices"]
    positive = p % 2 == 1
    row = np.where(positive, (p - 1) // 2, (p - 2) // 2)
    pos = {r: s[i] for i, r in enumerate(row) if positive[i]}
    neg = {r: s[i] for i, r in enumerate(row) if not positive[i]}
    order = list(dict.fromkeys(row.tolist()))         # first appearance
    complete = [r for r in order if r in pos and r in neg]
    y = np.array([pos[r] - neg[r] for r in complete])
    return y, np.array(complete, dtype=np.int64)
