"""On-disk format for experimentally measured single-pixel data.

ASPIR's pipeline normally simulates the acquisition: it projects patterns onto
dataset images to obtain the bucket values. Measured data enter *after* that
step. A measurement file carries everything a reconstruction needs:

``y``                 ``(F, M)`` float64 bucket values, one row per frame
                      (repeat) and one column per pattern, already reduced to
                      one number per pattern (for positive/negative DMD
                      acquisitions: positive minus negative).
``patterns``          ``(M, H, W)`` patterns, in the same order as the columns
                      of ``y``. Stored as int8 when they are +-1.
``img_size``          H (= W).
``acquisition_order`` ``(M,)`` position of each pattern in the instrument's own
                      acquisition sequence.
``reference``         optional ``(H, W)`` reference image (e.g. the digital
                      original of a printed target).
``flat_field``        optional ``(H, W)`` illumination image for flat-field
                      correction.
``metadata``          free-form dict (instrument, integration time, units,
                      licence, source DOI, processing options), stored as JSON.

Files are ``.npz`` with ``format = "aspir-measurements"`` and ``version = 1``.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simulation_engine._1_dataset_gen.dataset import DatasetABC

FORMAT_NAME = "aspir-measurements"
FORMAT_VERSION = 1


@dataclass
class MeasuredData:
    """Measured bucket values plus the patterns that produced them."""

    y: np.ndarray
    patterns: np.ndarray
    img_size: int
    acquisition_order: np.ndarray = None
    reference: np.ndarray = None
    flat_field: np.ndarray = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.y = np.atleast_2d(np.asarray(self.y, dtype=np.float64))
        self.patterns = np.asarray(self.patterns)
        if self.patterns.ndim != 3:
            raise ValueError(f"patterns must be (M, H, W), got {self.patterns.shape}")
        if self.y.shape[1] != self.patterns.shape[0]:
            raise ValueError(f"y has {self.y.shape[1]} columns but there are "
                             f"{self.patterns.shape[0]} patterns")
        if self.patterns.shape[1:] != (self.img_size, self.img_size):
            raise ValueError(f"patterns are {self.patterns.shape[1:]}, img_size is {self.img_size}")
        if self.acquisition_order is None:
            self.acquisition_order = np.arange(self.patterns.shape[0])
        self.acquisition_order = np.asarray(self.acquisition_order, dtype=np.int64)

    @property
    def n_patterns(self) -> int:
        return int(self.patterns.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.y.shape[0])

    def subset(self, rows, note=None) -> "MeasuredData":
        """The measurements of ``rows`` (pattern indices), in that order."""
        rows = np.asarray(rows, dtype=np.int64)
        meta = dict(self.metadata)
        if note:
            meta.setdefault("subsets", []).append(note)
        return MeasuredData(
            y=self.y[:, rows], patterns=self.patterns[rows], img_size=self.img_size,
            acquisition_order=self.acquisition_order[rows], reference=self.reference,
            flat_field=self.flat_field, metadata=meta,
        )

    def mean_frame(self) -> "MeasuredData":
        """Average the frames (repeats) into one."""
        out = self.subset(np.arange(self.n_patterns))
        out.y = self.y.mean(axis=0, keepdims=True)
        out.metadata["frames_averaged"] = self.n_frames
        return out


def save_measurements(path, data: MeasuredData, compress=True):
    """Write ``data`` to ``path`` (.npz)."""
    patterns = data.patterns
    if np.issubdtype(patterns.dtype, np.floating) and np.all(np.abs(patterns) == 1):
        patterns = patterns.astype(np.int8)
    arrays = {
        "format": np.array(FORMAT_NAME), "version": np.int64(FORMAT_VERSION),
        "y": data.y, "patterns": patterns, "img_size": np.int64(data.img_size),
        "acquisition_order": data.acquisition_order,
        "metadata": np.array(json.dumps(data.metadata, default=str)),
    }
    if data.reference is not None:
        arrays["reference"] = np.asarray(data.reference, dtype=np.float64)
    if data.flat_field is not None:
        arrays["flat_field"] = np.asarray(data.flat_field, dtype=np.float64)
    (np.savez_compressed if compress else np.savez)(path, **arrays)


def load_measurements(path) -> MeasuredData:
    """Read a file written by :func:`save_measurements`."""
    with np.load(path, allow_pickle=False) as npz:
        if "format" not in npz.files or str(npz["format"]) != FORMAT_NAME:
            raise ValueError(f"{path} is not an ASPIR measurement file")
        version = int(npz["version"])
        if version > FORMAT_VERSION:
            raise ValueError(f"{path} uses format version {version}; this ASPIR reads up to "
                             f"{FORMAT_VERSION}")
        return MeasuredData(
            y=npz["y"], patterns=npz["patterns"], img_size=int(npz["img_size"]),
            acquisition_order=npz["acquisition_order"],
            reference=npz["reference"] if "reference" in npz.files else None,
            flat_field=npz["flat_field"] if "flat_field" in npz.files else None,
            metadata=json.loads(str(npz["metadata"])),
        )


class DatasetFromMeasurements(DatasetABC):
    """A dataset whose acquisition already happened: it carries measured ``y``.

    ``data`` holds the reference image when there is one (otherwise a zero
    image), so code that sizes itself from ``dataset.data`` keeps working; it
    is *not* a ground truth. ``is_experimental`` tells the GUI to skip the steps
    that need one (network training, reports).
    """

    is_experimental = True

    def __init__(self, img_size=None, path=None, measurements=None, logger=None, name=None,
                 loader=None):
        """``path`` (an ASPIR measurement file), ``measurements`` (a MeasuredData) or
        ``loader`` (a callable returning one, run by :meth:`load_data`, e.g. a
        format adapter such as the SPIHIM one)."""
        if measurements is None and path is None and loader is None:
            raise ValueError("give a measurement file (path), a MeasuredData or a loader")
        self.path = path
        self.loader = loader
        self.measurements = measurements
        if img_size is None:
            img_size = measurements.img_size if measurements is not None else 0
        if name is None:
            name = os.path.splitext(os.path.basename(path))[0] if path else "measurements"
        super().__init__(name, img_size, logger)
        if measurements is not None:
            self._adopt(measurements)

    def _adopt(self, measurements):
        if self.img_size and measurements.img_size != self.img_size:
            raise ValueError(f"measurements are {measurements.img_size}x{measurements.img_size}, "
                             f"the dataset expects {self.img_size}")
        self.measurements = measurements
        self.img_size = measurements.img_size
        ref = measurements.reference
        image = (np.asarray(ref, dtype=np.float64) if ref is not None
                 else np.zeros((self.img_size, self.img_size)))
        self.data = [image]
        self.num_images = 1

    def load_data(self, progress_callback=None):
        if self.measurements is None:
            self.logger.info("Loading measurements: %s", self.path)
            self._adopt(self.loader() if self.loader is not None else load_measurements(self.path))
        if progress_callback:
            progress_callback(1, 1)
        return True

    def to_dataframe(self):
        return pd.DataFrame([img.flatten() for img in self.data])
