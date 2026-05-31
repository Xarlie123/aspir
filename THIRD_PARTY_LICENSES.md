# Third-Party Licenses

ASPIR is released under the Apache License 2.0 (see `LICENSE`). It depends
on the third-party Python packages listed below. ASPIR imports these as
unmodified, separately-installed dependencies (via `pip install`); it does
not embed, fork, or statically link their sources. The licenses are
reproduced/summarised here for convenience — the authoritative text ships
inside each installed package's distribution metadata.

Versions shown are the ones resolved at the time of writing (May 2026);
exact versions depend on your platform and `pip` resolution. Run
`pip show <package>` for the version actually installed in your environment.

## Permissive licenses (Apache / BSD / MIT / PSF / HPND)

These licenses are compatible with redistribution under Apache 2.0 and
impose only attribution-style obligations.

| Package                  | License                              |
|--------------------------|--------------------------------------|
| numpy                    | BSD-3-Clause                         |
| scipy                    | BSD-3-Clause                         |
| pandas                   | BSD-3-Clause                         |
| Pillow                   | MIT-CMU (HPND)                       |
| opencv-python-headless   | Apache-2.0                           |
| scikit-image             | BSD-3-Clause                         |
| matplotlib               | Matplotlib License (PSF-based, BSD-compatible) |
| torch                    | BSD-3-Clause                         |
| torchvision              | BSD-3-Clause                         |
| lpips                    | BSD-2-Clause                         |
| LightPipes               | MIT                                  |
| onnxscript               | MIT                                  |
| kaggle                   | Apache-2.0                           |
| PyYAML                   | MIT                                  |
| tqdm                     | MPL-2.0 AND MIT                      |
| pdf2image                | MIT                                  |
| psutil                   | BSD-3-Clause                         |
| pynvml                   | BSD-3-Clause                         |

## Weak-copyleft licenses (LGPL-3.0)

These are used strictly as dynamically-imported libraries. The LGPL permits
this in a non-(L)GPL application as long as the user can replace the library
with a modified version — which a normal `pip install <pkg>==<other-version>`
already allows. ASPIR ships no modified copy of any of them.

| Package           | License                                          | Notes |
|-------------------|--------------------------------------------------|-------|
| PySide6           | LGPL-3.0-only (offered also under GPL-2.0/GPL-3.0 by The Qt Company; we use it under LGPL-3.0) | Qt for Python GUI bindings |
| shiboken6         | LGPL-3.0-only (same multi-license as PySide6)    | PySide6 binding generator runtime |
| pylops            | LGPL-3.0                                          | FISTA / TV-norm iterative reconstruction |
| pyproximal        | LGPL-3.0                                          | Proximal operators for TV-norm |

## Optional dependency — Jetson only

The `[jetson]` extra pulls in `jetson-stats` (the `jtop` daemon/CLI), which
is licensed **AGPL-3.0**. ASPIR talks to it through the `jtop` Python client
to read GPU/energy telemetry on NVIDIA Jetson devices, and only on those
devices. It is an optional, install-on-demand dependency that is never
required to build, run, or redistribute the core x86/desktop application. If
you redistribute a Jetson-targeted bundle that includes `jetson-stats`, be
aware of the AGPL-3.0 network-use clause and treat it as a separately
licensed, separately installed component.

| Package        | License    | Notes |
|----------------|------------|-------|
| jetson-stats   | AGPL-3.0   | Optional `[jetson]` extra; runtime telemetry only |

## Bundled tools (Docker image, not Python packages)

The Docker image additionally installs system packages and tools (TeX Live,
Poppler, NVIDIA Nsight Systems CLI, Qt/X11 runtime libraries) under their
respective upstream licenses. These are standard distribution packages
installed via `apt`, not redistributed as part of the ASPIR source tree.
