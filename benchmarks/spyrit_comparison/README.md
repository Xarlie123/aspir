# Head-to-head with SPyRiT 3.1.1

Answers Reviewer 2 (ICOMPUTING-D-26-00705): a feature matrix, a reconstruction-quality parity check on the
CelebA / Cake-Cutting Hadamard task of Use Case 2, and a workflow-effort comparison.

SPyRiT is not an ASPIR dependency. It runs in its own virtual environment, and each tool is driven only
through its public API. ASPIR's source is untouched by this folder. Two ASPIR changes came out of the
comparison, each in its own commit:
- `403db38` fixes a bug found along the way (omitted `.batch_config` keys);
- `b83434f` adds the *Keep best-validation weights* option, after the learned-parity gap traced back to
  weight selection (below).

## Versions

| | |
|---|---|
| SPyRiT | 3.1.1 from PyPI (git tag `3.1.1`, commit `f64f5ef`), licence LGPL-3.0 |
| torch / CUDA | 2.6.0+cu124 / 12.4, in both environments |
| ASPIR | this repository, HEAD (Apache-2.0) |
| Hardware | NVIDIA GeForce GTX 1650 (4 GB), AMD Ryzen 7 4800H, Ubuntu (kernel 6.8) |

## Reproduce

```bash
cd benchmarks/spyrit_comparison
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install spyrit==3.1.1
ASPIR=../../.venv/bin/python                     # ASPIR's own environment

$ASPIR uc2_data.py                 # rebuild UC2 images + split; checks the 300 test images against UC2
.venv/bin/python spyrit_side.py linear
$ASPIR aspir_side.py linear        # (2a) linear parity
.venv/bin/python spyrit_side.py train
.venv/bin/python spyrit_side.py evaluate
$ASPIR aspir_side.py learned       # (2b) UC2 checkpoints, not retrained
$ASPIR aspir_side.py retrain       # (2b, supplementary) UC2 recipe with today's ASPIR, for training time
$ASPIR aspir_side.py retrain_bestval   # (2b, diagnostic) last-epoch vs best-validation weights, one run
$ASPIR aspir_side.py retrain_keepbest  # (2b) UC2 recipe with ASPIR's keep-best-validation option
$ASPIR compare.py                  # tables + figure, one scoring function for both tools
$ASPIR spihim_split.py             # (2c) split SPIHIM star measurements
.venv/bin/python pretrained.py     # (2c) SPyRiT's published DC-Net weights
$ASPIR context.py                  # (2c) scores
```

Data and weights land in `data/`, `models/` and `stat/`, which git ignores. The SPyRiT source checkout
(`_ext/`, used as evidence for the feature matrix) is ignored too.

## Files

| File | Content |
|---|---|
| `feature_matrix.{md,csv}` | (1) capabilities with the evidence for each cell |
| `parity_linear.{md,csv}` | (2a) noiseless linear parity, per ratio |
| `parity_learned.{md,csv}` | (2b) pseudoinverse + U-Net in both tools, per ratio |
| `parity_psnr.{pdf,png}` | PSNR vs M/N, both tools, linear and learned, mean ± std |
| `context_pretrained.{md,csv}`, `context_spihim.{pdf,png}` | (2c) SPyRiT's pretrained DC-Net, context only |
| `workflow_effort.md`, `workflow_spyrit.py`, `uc2_minimal.batch_config` | (3) the same experiment in both tools |

## How parity is made exact

- **Same images.** ASPIR's own `DatasetFromCelebrities` (seed 10, 3000 images, INT8) and `PostprocessorNN`'s
  split (`random_split`, seed 10) rebuild Use Case 2's data. The 300 test images equal the stored UC2
  `originals` exactly (max difference 0).
- **Same rows.** UC2's own `masks.npz` rows are mapped to SPyRiT's `HadamSplit2d` through an order matrix
  (`spyrit_common.order_for_rows`). `meas_op.H` then equals the UC2 masks row by row, in order and sign, at
  every ratio. Regenerating Cake-Cutting instead would differ at 4 % and 8 %, because tie order depends on the
  NumPy build.
- **Same conventions.** Images in [0, 1] (SPyRiT 3's convention, and UC2's), noiseless (UC2 simulated no
  measurement noise; SPyRiT's default noise model is the identity), split measurements with `prep.Unsplit`.
- **Same budget.** One network per ratio:
  - 2400 / 300 training / validation images;
  - 50 epochs, batch 16, Adam, lr 1e-3, weight decay 1e-4, MSE, constant learning rate;
  - U-Nets of similar size (UC2: 486 k parameters; SPyRiT `Unet`: 500 k).
- **Same metric code.** `compare.score()` for every output: clip to [0, 1]; PSNR and SSIM with data range 1;
  LPIPS (AlexNet).

Remaining differences, which are part of each tool and are not removed:
- SPyRiT's `train_model` keeps the best-validation weights. ASPIR keeps the last epoch by default, and Use
  Case 2 was trained that way; its *Keep best-validation weights* option (off by default) does what
  SPyRiT does.
- UC2 fed its network min-max-normalised, 8-bit-quantised linear reconstructions; SPyRiT feeds the
  pseudoinverse in [0, 1] directly.

## Results in brief

- **Linear (2a).** Identical to float32 precision at every ratio: max |difference| ≤ 3·10⁻⁶; PSNR and SSIM
  equal to four decimals. On the real SPIHIM star the two tools also agree exactly on the same 1024 rows.
- **Learned (2b), PSNR in dB on the 300 UC2 test images:**

  | M/N | ASPIR, UC2 checkpoints | SPyRiT PinvNet + Unet | ASPIR retrained, *Keep best-validation weights* (epoch) |
  |---|---|---|---|
  | 0.04 | 22.60 | 22.63 | 22.58 (38) |
  | 0.08 | 23.59 | 24.57 | 24.37 (45) |
  | 0.16 | 26.03 | 27.12 | 26.42 (31) |
  | 0.24 | 27.18 | 29.02 | 28.28 (48) |
  | 0.32 | 29.47 | 29.75 | 29.77 (45) |

  - **Configuration difference found.** SPyRiT's `train_model` returns the best-validation weights; ASPIR's
    training kept the last epoch, and Use Case 2 was trained that way.
    - With a constant learning rate the last epoch is noisy. Two identical ASPIR retrainings differ by up to
      1.4 dB at the last epoch (16 %: 26.28 vs 24.86 dB).
    - At 32 % the last epoch gives 26.1–26.4 dB against 29.7 dB at the best epoch of the same run.
  - **ASPIR now has the same rule** as an option (`b83434f`, off by default; `aspir_side.py
    retrain_keepbest`). With it, ASPIR is within 0.2 dB of SPyRiT at 4 %, 8 % and 32 %, and within 0.7 dB at
    16 % and 24 %.
    - The earlier diagnostic (`retrain_bestval`, best epoch picked outside ASPIR) gave the same picture: its
      values are within 0.2 dB of the option's at every ratio, where the two last-epoch runs differed by up
      to 1.4 dB.
    - Training time with the option: 2.0–2.3 min per net, against 2.0–2.2 without it.
  - Not attributed further (0.7 dB at 16 % and 24 %): the INT8 quantisation of the network input in the UC2
    recipe, the U-Net decoders (ASPIR's current U-Net has 541 k parameters), and run-to-run variance.
  - Timing on the GTX 1650:
    - training 2.0 min per net in ASPIR (including PSNR / SSIM / LPIPS on validation every epoch) against
      2.5 min in SPyRiT;
    - inference 2.2 ms per image in both (acquisition, linear reconstruction and U-Net).
- **Context (2c).** SPyRiT's published DC-Net + U-Net (STL-10, M = 1024, variance order) gives 17.87 dB on
  the SPIHIM star, against 17.01 dB for the linear reconstruction of the same rows. ASPIR's STL-10 U-Net gives
  16.60 dB at M = 983 Cake-Cutting (R1.3b). Different ordering, M and training: context, not parity.
