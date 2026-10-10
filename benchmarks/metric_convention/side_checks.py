#!/usr/bin/env python
"""Task F: parameter counts and exposure to the 403db38 bug (ASPIR venv).

    ../../.venv/bin/python side_checks.py

F1. Trainable parameters, sum(p.numel() for p in model.parameters() if p.requires_grad), for:
    the UC2 checkpoints of both stored runs and the UC1 checkpoints (UNetTranspose, their decoder predates
    2026-05-16; strict load), ASPIR's current UNet with the same configurations, and a per-block breakdown.
F2. Whether the stored UC1 / UC2 configurations and reports state the reconstruction method and the timing
    warm-up count (403db38: a hand-written .batch_config without them ran the pseudoinverse / 5 warm-ups).
Writes side_checks.md.
"""

import json
from collections import defaultdict

import torch
from common import BATCH, HERE, RUNS, md_table, report

from simulation_engine._4_postprocessor.models.unet import UNet
from simulation_engine._4_postprocessor.models.unet_transpose import UNetTranspose


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def blocks(model):
    out = defaultdict(int)
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name.split(".")[0]] += p.numel()
    return dict(out)


def checkpoint_rows():
    rows, breakdown = [], {}
    for run in RUNS:
        for t in report(run)["results"]:
            feats = t.get("architecture_config", {}).get("features")
            path = RUNS[run] / "models" / (t["name"].replace(" ", "_") + ".pt")
            state = torch.load(path, map_location="cpu")
            net = UNetTranspose(features=feats)
            net.load_state_dict(state)                       # strict: raises if the architecture differs
            rows.append({"run": run, "test": t["name"], "features": str(feats), "class": "UNetTranspose",
                         "trainable params": trainable(net),
                         "bottleneck channels": feats[-1] * 2})
            breakdown[str(feats)] = blocks(net)
    return rows, breakdown


def current_rows():
    rows = []
    for feats in ([8, 16, 32, 64], [8, 16]):
        net = UNet(features=feats)
        rows.append({"run": "current UNet (Upsample decoder)", "test": "-", "features": str(feats),
                     "class": "UNet", "trainable params": trainable(net), "bottleneck channels": feats[-1] * 2})
    return rows


def candidates_130k():
    """Configurations whose count is near 130 K, to locate where the manuscript figure could come from."""
    out = []
    for feats in ([8, 16, 32], [8, 16, 32, 64], [4, 8, 16, 32, 64], [16, 32, 64], [8, 16, 32, 48]):
        for cls in (UNetTranspose, UNet):
            n = cls(features=feats)
            enc = sum(v for k, v in blocks(n).items() if k in ("downs", "encoder", "enc"))
            out.append({"class": cls.__name__, "features": str(feats), "trainable params": trainable(n),
                        "encoder only": enc, "without bottleneck": trainable(n) - blocks(n).get("bottleneck", 0)})
    return out


def config_rows():
    rows = []
    for f in sorted(BATCH.glob("uc*_*.batch_config")):
        cfg = json.loads(f.read_text())
        for t in cfg["tests"]:
            rows.append({"source": f.name, "test": t["name"],
                         "reconstruction_method": t.get("reconstruction_method", "MISSING"),
                         "timing_warmup_runs": t.get("timing_warmup_runs", "MISSING"),
                         "saved by": "GUI" if "format_version" in cfg else "?"})
    for run in RUNS:
        for name in sorted(p.name for p in RUNS[run].glob("*.batch_analysis_report")):
            rep = json.loads((RUNS[run] / name).read_text())
            for t in rep["results"]:
                rows.append({"source": f"{RUNS[run].name}/{name}", "test": t["name"],
                             "reconstruction_method": t.get("reconstruction_method", "MISSING"),
                             "timing_warmup_runs": t.get("timing_warmup_runs",
                                                         t.get("timing_results", {}).get("warmup_runs", "not stored")),
                             "saved by": rep["metadata"].get("remeasured_device", "batch run")})
    return rows


def main():
    ck, breakdown = checkpoint_rows()
    md = ["# Task F — side checks", "", "## F1. Trainable parameters", "",
          md_table(ck + current_rows(), ["run", "test", "features", "class", "trainable params",
                                        "bottleneck channels"]), "",
          "Per-block breakdown of the stored checkpoints (UNetTranspose):", ""]
    for feats, b in breakdown.items():
        md.append(f"- features {feats}: " + ", ".join(f"{k} {v:,}" for k, v in b.items()))
    md += ["", "Candidate configurations near the manuscript's 130 K:", "",
           md_table(candidates_130k(), ["class", "features", "trainable params", "encoder only",
                                        "without bottleneck"]), "",
           "## F2. Exposure to the 403db38 bug", "",
           md_table(config_rows(), ["source", "test", "reconstruction_method", "timing_warmup_runs", "saved by"]),
           ""]
    (HERE / "side_checks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
