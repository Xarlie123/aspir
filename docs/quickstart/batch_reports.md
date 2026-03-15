# Batch Reports

Batch Reports mode allows you to explore offline the results of previously executed batch experiments.

Since training multiple experiments can be time-consuming, this mode is useful for loading and analyzing data from experiments that have already been trained, without having to re-run them.

## Loading Experiments

1. Switch to **Batch Reports** mode
2. Click **Load Experiment** and select your experiment folder
3. Multiple experiments can be loaded for comparison

## Available Views

- **Summary**: Table with all tests and their metrics
- **Quality**: Bar charts comparing PSNR, SSIM, LPIPS across tests
- **Timing**: Pipeline latency breakdown per stage
- **Training**: Loss curves and convergence analysis
- **Details**: Full configuration and per-test metrics

## Exporting Reports

Results can be exported to HTML, PDF, LaTeX, or CSV.
