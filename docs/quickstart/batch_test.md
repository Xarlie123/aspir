# Batch Test Pipeline

The Batch Test mode allows you to define multiple experiments and run them automatically, making it easy to compare different configurations.

```{mermaid}
flowchart LR
    A[Configure Tests] --> B[Run Batch]
    B --> C[Load Results]
    C --> D[Compare & Export]
```

```{toctree}
:maxdepth: 1
:caption: Batch Test Steps

batch_step1_configure
batch_step2_run
batch_step3_results
```
