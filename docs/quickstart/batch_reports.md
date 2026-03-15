# Batch Reports

```{note}
Since training multiple experiments can be time-consuming, Batch Reports mode allows you to explore offline the results of previously executed batch experiments, loading and analyzing data without having to re-run them.
```

```{raw} html
<div style="max-width: 350px;">
```

```{mermaid}
flowchart LR
    A[Load Batch Results] --> B[Compare & Export]
```

```{raw} html
</div>
```

Load `.batch_analysis_report` files to review completed experiments. You can also load multiple report files to compare results across different batch experiments.

You can also load multiple report files to compare results across different batch experiments.

## Loading Experiments

1. Switch to **Batch Reports** mode
2. Click **Load Experiment** and select a `.batch_analysis_report` file
3. To compare across experiments, load additional report files

## Available Views

- **Summary**: Overview table with all tests and their metrics, color-coded for easy comparison
- **Quality**: Bar charts comparing PSNR, SSIM, and LPIPS across tests, with global normalization for fair comparison
- **Timing**: Pipeline latency breakdown per stage (acquisition, reconstruction, inference)
- **Training**: Loss curves and convergence analysis for each neural network model
- **Details**: Full configuration dump and per-test metrics

## Exporting Reports

Results can be exported to HTML, PDF, LaTeX, or CSV for use in publications or further analysis.

```{raw} html
<video width="80%" controls>
  <source src="../1_batch_reports_from_file.mp4" type="video/mp4">
</video>
```
