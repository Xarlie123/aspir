# Workflow effort: the same experiment in SPyRiT 3.1.1 and in ASPIR

Experiment: CelebA 64×64, Cake-Cutting Hadamard, M/N ∈ {0.04, 0.08, 0.16, 0.24, 0.32}, pseudoinverse +
U-Net, PSNR / SSIM / timing per ratio. Extra row: "… plus energy per image".

**Bias, stated openly.** The SPyRiT script was written by someone who knows ASPIR far better than SPyRiT, by
following SPyRiT's tutorials (`tuto_01_c`, `tuto_03`, `tuto_04_b`, `tuto_05`). It is kept idiomatic and
minimal, not padded. It was run end to end (smoke run: 1 epoch, 64 images) to make sure it works.

## SPyRiT: `workflow_spyrit.py`

- **55 lines** of code (non-blank, non-comment, no docstrings), one file. It does the whole experiment: load
  CelebA, Cake-Cutting order, five ratios, train PinvNet + Unet, evaluate, time.
- **Concepts a user must know:**
  - `HadamSplit2d` and its `order` matrix: higher value = measured first; stable sort.
  - `define_order(n, "CC", pdf=True)`. With `pdf=False` it returns ranks (0 = best), which passed as `order`
    select the *least* significant patterns: 0 of the Cake-Cutting rows. See the pitfall below.
  - Split measurement and `prep.Unsplit` (or `UnsplitRescale(alpha)` with Poisson noise).
  - `PinvNet` / `FullNet` (`acquire`, `reconstruct`, `reconstruct_pinv`); `nnet.Unet`.
  - `train_model`:
    - a dict of loaders yielding `(input, label)` pairs;
    - a criterion called as `criterion(x, y, net)` (hence `Weight_Decay_Loss`);
    - a required scheduler;
    - it keeps the best-validation weights.
  - `save_net` / `load_net`.
  - Image range: SPyRiT 3 works on [0, 1]; `transform_gray_norm(normalize=False)`. SPyRiT 2.4 weights expect
    [−1, 1] and need `prep.Rerange`.
  - Metrics:
    - `batch_psnr` uses each image's own range;
    - `batch_psnr_(…, r=1)` uses a fixed range;
    - `batch_ssim` is a global, non-windowed SSIM;
    - all three return per-image lists.
  - CUDA synchronisation for timing.
- **Provided by SPyRiT, not written:**
  - the Cake-Cutting order (`define_order 'CC'`: same rows as ASPIR at 8–32 %, 161 / 163 at 4 %, tie order);
  - operators, pseudoinverse, U-Net, training loop, PSNR / SSIM.

  CelebA needs no dedicated loader: `torchvision.datasets.ImageFolder` reads the Kaggle folder.
- **Written from scratch:**
  - subset of 3000 images and the 80 / 10 / 10 split;
  - per-ratio loop and result table;
  - averaging of the per-image metric lists;
  - GPU-synchronised per-image timing with warm-up.
- **Energy per image (extra row): nothing in SPyRiT.**
  - A minimal NVML before/after measurement is about 20 lines.
  - Doing what ASPIR does is ~1,070 lines in ASPIR: NVML and RAPL backends, idle baseline, per-stage
    backend attribution, propagated σ and bootstrap interval (`_5_analyzer/energy_backends`,
    `baseline_capture.py`, `analyzer_energy.py`).
- **Outputs:** the printed table and the `.pth` weights (TensorBoard optional).

## ASPIR: Batch Test GUI

Actions from a started application, assuming Kaggle credentials are configured (otherwise two more fields).
Every value not listed is already the GUI default and equals Use Case 2: U-Net [8, 16, 32, 64], 50 epochs,
batch 16, lr 1e-3, weight decay 1e-4, MSE, Adam, split 80 / 10 / 10, reports incl. timing and energy, idle
baseline, seed 10.

| # | Action | Count |
|---|---|---|
| 1 | Dataset step: *Download from Internet* | 1 |
| 2 | Dataset *Celebrities*, size 64, 3000 images, data format INT8 | 4 |
| 3 | *Generate Dataset* (plus *Download Dataset* the first time CelebA is used) | 1–2 |
| 4 | Switch to *Batch Test* | 1 |
| 5 | *+ Add*; set name, mask type *hadamard_cake_cutting*, percentage 4 % | 4 |
| 6 | *Duplicate* + name + percentage, for 8 / 16 / 24 / 32 % | 12 |
| 7 | Batch name, *Run All Tests* | 2 |
| 8 | Switch to *Batch Reports*, *Load Executed Batch Test* | 2 |
| | **Total** | **27–28** (23–24 if tests keep their default names) |

- **Energy per image:** 0 extra actions (on by default).
- **Scripted user:** `uc2_minimal.batch_config` is **13 lines** of JSON. It gives the same five tests as Use
  Case 2's 330-line GUI-saved config (all omitted keys take the GUI defaults; checked field by field after
  fixing `from_dict`, ASPIR commit `403db38`).
  - There is no batch CLI. The config is loaded in the GUI (*Load Batch Config*, *Run All Tests*: 2 actions)
    or driven from Python through `BatchTestRunner` (about 30 lines, as in this comparison's e2e check).
- **Outputs:**
  - a `.batch_analysis_report` (JSON) per batch, with per test: PSNR / SSIM / LPIPS mean ± std, training
    curves, GPU-synchronised timing of reconstruction and inference, energy per stage with idle baseline,
    dynamic energy and σ;
  - exported models (`.pt`, `.onnx`), test images and masks;
  - comparison charts and figure export in *Batch Reports*.

## Install and machine time (same desktop: GTX 1650, Ryzen 7 4800H)

| | SPyRiT 3.1.1 | ASPIR (HEAD) |
|---|---|---|
| Fresh venv + `pip install` (pip cache warm, torch not downloaded) | 114 s, 53 packages | 184 s, 99 packages (includes the Qt GUI) |
| Training, five nets, 50 epochs, 2400 images | 12.8 min (2.5–2.6 min per net) | 10.2 min (2.0–2.2 min per net), including PSNR / SSIM / LPIPS on validation every epoch |
| Extra measurements per test | none | timing (800 runs) and energy (800 runs + idle baseline) |

What the numbers support: SPyRiT reaches the result with a short, readable script for a Python user who knows
its concepts. ASPIR reaches it with ~25 GUI actions and no code, and adds timing, energy and the report
without extra effort. The two serve different users; neither number is an effort "score".
