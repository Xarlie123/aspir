# Batch Test Pipeline

The Batch Test mode allows you to define multiple experiments and run them automatically, making it easy to compare different configurations.

```{mermaid}
flowchart LR
    A[Configure Tests] --> B[Run Batch]
    B --> C[Load Results]
    C --> D[Compare & Export]
```

## Step 1: Create Test Configurations

1. Switch to **Batch Test** mode
2. Click **Add Test** to create a new configuration
3. For each test, configure:
   - **Mask**: Type, number of patterns, seed
   - **Reconstruction**: Method (Ghost Imaging, Pseudoinverse, FISTA, TV-Norm)
   - **Model**: Architecture, epochs, batch size, learning rate
4. Repeat to add as many configurations as needed

## Step 2: Run the Batch

1. Set an **Export Name** to identify your results
2. Click **Run Batch**
3. Monitor progress in the status panel

Tests run sequentially by default. Parallel execution is available for independent tests.

## Step 3: Explore Results

1. Switch to **Batch Reports** mode
2. Click **Load Experiment** and select your experiment folder
3. Browse the available views:
   - **Summary**: Table with all tests and their metrics
   - **Quality**: Bar charts comparing PSNR, SSIM, LPIPS across tests
   - **Timing**: Pipeline latency breakdown per stage
   - **Training**: Loss curves and convergence analysis
   - **Details**: Full configuration and per-test metrics

Results can be exported to HTML, PDF, LaTeX, or CSV.
