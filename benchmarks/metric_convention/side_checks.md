# Task F — side checks

## F1. Trainable parameters

| run | test | features | class | trainable params | bottleneck channels |
|---|---|---|---|---|---|
| UC2_R0 | Celeb - Cake Cutting 4% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_R0 | Celeb - Cake Cutting 8% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_R0 | Celeb - Cake Cutting 16% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_R0 | Celeb - Cake Cutting 24% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_R0 | Celeb - Cake Cutting 32% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_RERUN | Celeb - Cake Cutting 4% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_RERUN | Celeb - Cake Cutting 8% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_RERUN | Celeb - Cake Cutting 16% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_RERUN | Celeb - Cake Cutting 24% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC2_RERUN | Celeb - Cake Cutting 32% | [8, 16, 32, 64] | UNetTranspose | 486409 | 128 |
| UC1_SWEEP | Beam prof - stride 2 | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_SWEEP | Beam prof - stride 4 | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_SWEEP | Beam prof - stride 6 | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_SWEEP | Beam prof - stride 8 | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_CCH | Beam prof - CCH 2.2% | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_CCH | Beam prof - CCH 2.8% | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_CCH | Beam prof - CCH 4.2% | [8, 16] | UNetTranspose | 29481 | 32 |
| UC1_CCH | Beam prof - CCH 8.3% | [8, 16] | UNetTranspose | 29481 | 32 |
| current UNet (Upsample decoder) | - | [8, 16, 32, 64] | UNet | 540809 | 128 |
| current UNet (Upsample decoder) | - | [8, 16] | UNet | 32681 | 32 |

Per-block breakdown of the stored checkpoints (UNetTranspose):

- features [8, 16, 32, 64]: downs 73,704, bottleneck 221,696, ups 43,640, up_convs 147,360, final_conv 9
- features [8, 16]: downs 4,200, bottleneck 13,952, ups 2,584, up_convs 8,736, final_conv 9

Candidate configurations near the manuscript's 130 K:

| class | features | trainable params | encoder only | without bottleneck |
|---|---|---|---|---|
| UNetTranspose | [8, 16, 32] | 121033 | 18152 | 65481 |
| UNet | [8, 16, 32] | 134473 | 18152 | 78921 |
| UNetTranspose | [8, 16, 32, 64] | 486409 | 73704 | 264713 |
| UNet | [8, 16, 32, 64] | 540809 | 73704 | 319113 |
| UNetTranspose | [4, 8, 16, 32, 64] | 487397 | 74116 | 265701 |
| UNet | [4, 8, 16, 32, 64] | 541957 | 74116 | 320261 |
| UNetTranspose | [16, 32, 64] | 482449 | 72016 | 260753 |
| UNet | [16, 32, 64] | 536209 | 72016 | 314513 |
| UNetTranspose | [8, 16, 32, 48] | 303865 | 52904 | 179065 |
| UNet | [8, 16, 32, 48] | 337785 | 52904 | 212985 |

## F2. Exposure to the 403db38 bug

| source | test | reconstruction_method | timing_warmup_runs | saved by |
|---|---|---|---|---|
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results.batch_config | Beam prof - CCH 2.2% | conventional | 20 | GUI |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results.batch_config | Beam prof - CCH 2.8% | conventional | 20 | GUI |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results.batch_config | Beam prof - CCH 4.2% | conventional | 20 | GUI |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results.batch_config | Beam prof - CCH 8.3% | conventional | 20 | GUI |
| uc1_batch_test_strides_paper_2000.batch_config | Beam prof - stride 2 | conventional | 20 | GUI |
| uc1_batch_test_strides_paper_2000.batch_config | Beam prof - stride 4 | conventional | 20 | GUI |
| uc1_batch_test_strides_paper_2000.batch_config | Beam prof - stride 6 | conventional | 20 | GUI |
| uc1_batch_test_strides_paper_2000.batch_config | Beam prof - stride 8 | conventional | 20 | GUI |
| uc2_batch_test_celeb_3000_hadamard.batch_config | Celeb - Cake Cutting 4% | conventional | 20 | GUI |
| uc2_batch_test_celeb_3000_hadamard.batch_config | Celeb - Cake Cutting 8% | conventional | 20 | GUI |
| uc2_batch_test_celeb_3000_hadamard.batch_config | Celeb - Cake Cutting 16% | conventional | 20 | GUI |
| uc2_batch_test_celeb_3000_hadamard.batch_config | Celeb - Cake Cutting 24% | conventional | 20 | GUI |
| uc2_batch_test_celeb_3000_hadamard.batch_config | Celeb - Cake Cutting 32% | conventional | 20 | GUI |
| batch_test_celeb_hadamard_results_1/results.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 5 | batch run |
| batch_test_celeb_hadamard_results_1/results.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 5 | batch run |
| batch_test_celeb_hadamard_results_1/results.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 5 | batch run |
| batch_test_celeb_hadamard_results_1/results.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 5 | batch run |
| batch_test_celeb_hadamard_results_1/results.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 5 | batch run |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-cpu_20260502-1152.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | jetson-cpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-cpu_20260502-1152.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | jetson-cpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-cpu_20260502-1152.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | jetson-cpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-cpu_20260502-1152.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | jetson-cpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-cpu_20260502-1152.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | jetson-cpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-gpu_20260502-1200.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | jetson-gpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-gpu_20260502-1200.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | jetson-gpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-gpu_20260502-1200.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | jetson-gpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-gpu_20260502-1200.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | jetson-gpu |
| batch_test_celeb_hadamard_results_1/results_reexecuted_jetson-gpu_20260502-1200.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | jetson-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | batch run |
| uc2_batch_test_celeb_3000_hadamard_results/results.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | batch run |
| uc2_batch_test_celeb_3000_hadamard_results/results.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | batch run |
| uc2_batch_test_celeb_3000_hadamard_results/results.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | batch run |
| uc2_batch_test_celeb_3000_hadamard_results/results.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | batch run |
| uc2_batch_test_celeb_3000_hadamard_results/results_r13c_16pct.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | batch run |
| uc2_batch_test_celeb_3000_hadamard_results/results_r13c_16pct_reexecuted_desktop-cpu_20261010-1123.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | desktop-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_r13c_16pct_reexecuted_desktop-gpu_20261010-1121.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | desktop-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-cpu_20260502-1408.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | jetson_batch16-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-cpu_20260502-1408.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | jetson_batch16-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-cpu_20260502-1408.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | jetson_batch16-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-cpu_20260502-1408.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | jetson_batch16-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-cpu_20260502-1408.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | jetson_batch16-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-gpu_20260502-1514.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | jetson_batch16-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-gpu_20260502-1514.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | jetson_batch16-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-gpu_20260502-1514.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | jetson_batch16-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-gpu_20260502-1514.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | jetson_batch16-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch16-gpu_20260502-1514.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | jetson_batch16-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-cpu_20260502-1323.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | jetson_batch4-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-cpu_20260502-1323.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | jetson_batch4-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-cpu_20260502-1323.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | jetson_batch4-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-cpu_20260502-1323.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | jetson_batch4-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-cpu_20260502-1323.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | jetson_batch4-cpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-gpu_20260502-1348.batch_analysis_report | Celeb - Cake Cutting 4% | Hadamard Linear | 20 | jetson_batch4-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-gpu_20260502-1348.batch_analysis_report | Celeb - Cake Cutting 8% | Hadamard Linear | 20 | jetson_batch4-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-gpu_20260502-1348.batch_analysis_report | Celeb - Cake Cutting 16% | Hadamard Linear | 20 | jetson_batch4-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-gpu_20260502-1348.batch_analysis_report | Celeb - Cake Cutting 24% | Hadamard Linear | 20 | jetson_batch4-gpu |
| uc2_batch_test_celeb_3000_hadamard_results/results_reexecuted_jetson_batch4-gpu_20260502-1348.batch_analysis_report | Celeb - Cake Cutting 32% | Hadamard Linear | 20 | jetson_batch4-gpu |
| uc1_batch_test_strides_paper_2000_results/results.batch_analysis_report | Beam prof - stride 2 | Sweep Linear | 20 | batch run |
| uc1_batch_test_strides_paper_2000_results/results.batch_analysis_report | Beam prof - stride 4 | Sweep Linear | 20 | batch run |
| uc1_batch_test_strides_paper_2000_results/results.batch_analysis_report | Beam prof - stride 6 | Sweep Linear | 20 | batch run |
| uc1_batch_test_strides_paper_2000_results/results.batch_analysis_report | Beam prof - stride 8 | Sweep Linear | 20 | batch run |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results/results.batch_analysis_report | Beam prof - CCH 2.2% | Hadamard Linear | 20 | batch run |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results/results.batch_analysis_report | Beam prof - CCH 2.8% | Hadamard Linear | 20 | batch run |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results/results.batch_analysis_report | Beam prof - CCH 4.2% | Hadamard Linear | 20 | batch run |
| uc1_batch_test_cake_cutting_hadamard_paper_2000_results/results.batch_analysis_report | Beam prof - CCH 8.3% | Hadamard Linear | 20 | batch run |

