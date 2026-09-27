# Experimental Data

ASPIR normally *simulates* the acquisition: it projects the mask patterns onto
dataset images to obtain the single-pixel (bucket) values, then reconstructs.
Measured data enter after that step. A measurement file carries the bucket
values recorded on real hardware together with the patterns that produced
them, and every reconstruction algorithm runs on them unchanged.

## Measurement file format

Measurement files are `.npz` archives with `format = "aspir-measurements"` and
`version = 1` (`simulation_engine/_1_dataset_gen/measured.py`):

| key | shape | content |
|-----|-------|---------|
| `y` | `(F, M)` | bucket values, one row per frame (repeat), one column per pattern; already one number per pattern (for positive/negative DMD acquisitions: positive minus negative) |
| `patterns` | `(M, H, W)` | the patterns, in the column order of `y`; stored as `int8` when they are +-1 |
| `img_size` | scalar | `H` (= `W`) |
| `acquisition_order` | `(M,)` | position of each pattern in the instrument's own sequence |
| `reference` | `(H, W)`, optional | reference image, e.g. the digital original of a printed target |
| `flat_field` | `(H, W)`, optional | illumination image for flat-field correction |
| `metadata` | JSON string | instrument, integration time, units, licence, source DOI, processing options |

```python
from simulation_engine._1_dataset_gen.measured import MeasuredData, save_measurements, load_measurements

save_measurements("scene.npz", MeasuredData(y=y, patterns=patterns, img_size=64,
                                            metadata={"instrument": "...", "units": "counts"}))
data = load_measurements("scene.npz")
```

## Reconstructing measured values

Every applicator has `reconstruct(y, idx_mask_min=0, idx_mask_max=None)`. It
runs the same solver as the simulated path, which computes `y` itself and then
calls that step. Both devices are supported; the legacy NumPy backend only
reproduces the simulated pipeline and raises `NotImplementedError`.

`simulation_engine/experimental.py` wires this up:

```python
from simulation_engine import experimental as ex

x = ex.reconstruct_measured(data, method="native", device="cuda")   # (F, H, W)
sub = ex.subsample(data, "cake_cutting", 0.16)                      # ASPIR's first floor(0.16 N) rows
x_sim = ex.reconstruct_simulated(scene, sub, method="fista")        # same rows, simulated
```

- `method` is one of `"native"` (Hadamard Linear for +-1 patterns),
  `"pseudoinverse"`, `"fista"`, `"tv_norm"`.
- +-1 Sylvester Hadamard patterns are recognised whatever their order, and the
  reconstruction then uses the fast Walsh-Hadamard transform.
- **Sub-sampling in ASPIR's orderings.** From a full Hadamard acquisition,
  `subsample(data, ordering, ratio)` selects the first `floor(ratio * N)` rows
  of ASPIR's `natural`, `walsh_paley`, `cake_cutting` or `cal_sal` order,
  matching each measured pattern to ASPIR's by its Sylvester row index, and
  fixes the sign so the patterns are exactly ASPIR's. `"acquisition"` keeps the
  instrument's order, which also serves partial acquisitions.
  `subsample_as(data, patterns)` reproduces a given pattern set, for example
  the `masks.npz` a network was trained with.
- **Scale.** Measured values are in instrument units (SPIHIM: about 1e7
  counts). The FISTA and TV default weights were chosen for scenes in
  [0, 1] and have no effect on raw counts. `scale_to_unit_peak(data)` divides
  `y` by one constant so that the frame-averaged image peaks at 1, using the
  measurements only.

```{note}
ASPIR's Cake-Cutting order sorts patterns by block count with
`numpy.argsort`, which does not fix the order of equal counts. Two NumPy
builds can therefore pick different rows among ties at the cut-off. Use
`subsample_as` with the stored `masks.npz` when the exact training rows
matter.
```

## SPIHIM adapter

SPIHIM is the open experimental single-pixel collection of the OpenSpyrit
ecosystem. It was acquired with a Vialux V-7001 DMD, a BWTek Exemplar
spectrometer (2048 bins) behind a fibre, a white LED with an ND filter wheel
and a telecentric lens.

**Licence and citation.** The data are distributed under CC-BY 4.0. Work that
uses them must cite:

> G. Beneti Martins, L. Mahieu-Williame, T. Baudier, N. Ducros,
> "OpenSpyrit: an ecosystem for open single-pixel hyperspectral imaging",
> *Opt. Express* **31**(10), 15599 (2023), DOI 10.1364/OE.483937.

Download (about 1 GB, resumable, SHA-256 checked, into the git-ignored
`datasets/spihim/`):

```bash
python scripts/download_spihim.py              # pattern bank + 2020-Jun-11 + 2020-Nov-18
python scripts/download_spihim.py --with-jul   # also 2020-Jul-01 (STL-10 acquisitions)
```

```python
from simulation_engine._1_dataset_gen.spihim import HadamardPngBank, load_spihim

bank = HadamardPngBank("datasets/spihim/Hadamard_64x64")
data = load_spihim("datasets/spihim/spihim_2020-Jun-11/R1L1S2P_1_0.0_raw.mat", bank)
```

`load_spihim(mat_path, bank, spectral="sum", band=None, bin_index=None,
orientation=None, exclude_saturated=True, exclude_bins=None,
flat_field_mat=None, reference=None)`:

- Each `*_raw.mat` holds one spectrum per projected pattern (`M`,
  `(n_patterns, 2048)` uint16) and the 1-based index of each pattern in the
  8192-PNG bank. PNG `2k+1` / `2k+2` are the positive / negative halves of
  bipolar row `k`. `y[k]` is their difference, so the spectrometer's dark
  offset (about 1150 counts per bin) cancels.
- `spectral`: `"sum"` (panchromatic, the default), `"band"` with
  `band=(lo_nm, hi_nm)`, or `"bin"` with `bin_index`. The white LED puts
  nearly all the light between about 450 and 700 nm. The other bins add
  noise and a small positive/negative drift but no signal, so a band such as
  `(420, 720)` improves the single-shot signal-to-noise ratio (by about
  1.8 dB on the ten star repeats) and should be preferred for low-light
  acquisitions.
- **Saturation.** Some bins reach the uint16 ceiling (65535) in a few patterns
  of the brightest acquisitions. A clipped bin no longer measures light
  linearly, so it is dropped from the whole acquisition by default. For a
  series of repeats, pass the union as `exclude_bins` so that every repeat uses
  the same bins.
- **Orientation.** The bank's patterns are rotated by two quarter turns
  (`DEFAULT_ORIENTATION = {"rot90": 2, "flip": False}`). Of the eight
  rotations and mirrors of a full reconstruction of `R1L1S2P_1_0.0`, only the
  four unmirrored ones read the printed text correctly, and this one is the
  closest to upright. Rotating or mirroring a Sylvester pattern gives another
  +-Sylvester pattern, so the fast transform still applies.
- `flat_field_mat`: a `noObject*` acquisition of the same session and optical
  density. Its full linear reconstruction is stored as `flat_field`.

## In the GUI

In **Single Test**, the dataset step offers **Experimental measurements
(file)** for ASPIR `.npz` files and SPIHIM `*_raw.mat` acquisitions. The
SPIHIM pattern bank is found next to the session folder or asked for. Options:

- average repeated frames;
- scale to a unit-peak image (on by default, for the reasons above).

After loading:

- The mask step shows the loaded patterns read-only, with a
  reconstruction-method selector, and no acquisition is simulated.
- **Test Masks** reconstructs from the measured values; the slider selects how
  many of the measured patterns to use.
- **Post-processing** and **Reports** need ground-truth images to train and
  evaluate, so they are closed for measured data.
  `scripts/experimental_spihim_figure.py` shows how to apply a pre-trained
  network to measured reconstructions.

Batch Test does not accept measured data yet. It would need:

- a dataset entry that points to a measurement file;
- test entries that choose a sub-sampling order and ratio instead of
  generating masks;
- reports that tolerate a missing ground truth, or that take the file's
  reference image with the metric definitions made explicit.

## Reproducing the experimental validation

```bash
python scripts/download_spihim.py
python scripts/experimental_spihim_figure.py     # writes experiments/experimental_spihim/
```

`python scripts/train_stl10_unets.py` (run before the figure script) trains
the same U-Nets on STL-10 instead of CelebA, with the Use Case 2 recipe. It
streams only the STL-10 train split, about 0.4 GB, into the git-ignored
`datasets/stl10/`, and the figure script adds those networks as extra
columns.

The script runs the Cake-Cutting sub-sampling at M/N = 0.04 ... 0.32 of the
star targets through Hadamard Linear, FISTA, TV and the Use Case 2 U-Nets. It
also runs ASPIR's simulated acquisition of the same scene, a simulation with
noise calibrated on the ten repeats, and an optical-density sweep. Its module
docstring lists every choice the numbers depend on: reference, affine fit
before PSNR / SSIM, contrast threshold, and the network input normalisation.
