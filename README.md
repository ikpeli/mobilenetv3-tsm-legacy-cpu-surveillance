# Real-Time Suspicious-Activity Detection on Legacy CPU Hardware

Code, manifests, and results for the paper *Real-Time Suspicious-Activity Detection on Legacy
CPU Hardware Using MobileNetV3-Large with Temporal Shift Modeling*.

A MobileNetV3-Large backbone with a Temporal Shift Module, trained on a 500-video binary
subset of UCF-Crime and deployed on an Intel Core i5-4570 desktop with no GPU, reading an
RTSP sub-stream from an eight-channel NVR and dispatching alerts over Telegram and e-mail.

## Verifying the reported numbers

Every detection, event-level, calibration, and operating-point figure in Section IV is
recomputed from the exported per-window probabilities by:

```bash
pip install pandas numpy scikit-learn
python verify_paper_numbers.py --results .
```

It prints PASS or FAIL for each check against the value printed in the paper and exits
non-zero if any check fails. `verify_numbers_colab.ipynb` is the same thing packaged for
Google Colab, with the script embedded so nothing needs installing.

## Contents

### Manifests

| File | Contents |
|---|---|
| `full_manifest_500.csv` | All 500 videos with class, binary label, and assigned partition |
| `train_manifest.csv` | 350 videos: 175 Normal, 70 Burglary, 70 Stealing, 35 Fighting |
| `validation_manifest.csv` | 75 videos: 38 Normal, 15 Burglary, 15 Stealing, 7 Fighting |
| `test_manifest.csv` | 75 videos: 37 Normal, 15 Burglary, 15 Stealing, 8 Fighting |
| `SHA256SUMS.txt` | Checksums for the four manifests |
| `expanded_normal_manifest.csv` | The manifest as emitted by the notebook, before the split column was added |

Partitioning is at the video level, not the clip level, so no source video contributes
clips to more than one partition. No file from the UCF-Crime testing-normal directory
enters any manifest.

### Results

| File | Contents |
|---|---|
| `test_clip_probs.csv` | 7,425 per-window probabilities, test partition |
| `val_clip_probs.csv` | 8,561 per-window probabilities, validation partition |
| `test_video_level_scores.csv`, `val_video_level_scores.csv` | Per-video aggregated scores |
| `event_predictions.csv` | Event-level outcomes at the deployed operating point, test |
| `event_predictions_val_validation_tuned.csv` | The same on validation |
| `alert_sweep_validation.csv` | Operating-point sweep over smoothing factor, threshold, and confirmation count |
| `calibration_metrics.json` | Bin-wise calibration statistics |
| `final_summary.json` | Metrics emitted by the notebook |
| `deployment_benchmark.json`, `openvino_benchmark.json` | Latency and throughput on the target and development CPUs |
| `history.json` | Per-epoch training history |
| `motion_cache.json` | Cached background-subtraction motion scores used for hard-negative sampling |

**Note on metric naming.** Keys containing `single_clip_per_video` cover one uniformly
sampled clip per video, 75 clips per partition, matching the evaluation DataLoader. They are
*not* computed over the full sliding-window sequence. For that, use `test_clip_probs.csv`,
which holds every window the deployed service would score.

### Model and configuration

`model.onnx` is the exported graph. `model.xml` and `model.bin` are the FP32 OpenVINO
intermediate representation actually run on the deployment machine, and together they are
sufficient to reproduce the inference, latency, and detection results reported in the paper.
`model_int8.xml` and `model_int8.bin` are the INT8 variant evaluated in Section III-F and
rejected, included so that the reported accuracy cost can be checked.

The PyTorch training checkpoint (`best_prauc.pt`, epoch 13, validation PR-AUC 0.846) is not
included here because of file-size limits. It is needed only to resume or fine-tune training,
not to reproduce any reported result, and is available on request from the corresponding
author. `config.yaml`, `model_configuration.yaml`, and
`deployment_settings.yaml` carry the training, model, and pipeline settings.
`requirements.txt` pins the environment.

### Figures

`fig1_block_diagram.pdf` through `fig5_reliability.pdf` are the manuscript figures as vector
PDFs. `evaluation_plots.png`, `calibration_diagram.png`, and
`validation_fighting_recall_tradeoff.png` are the notebook's own diagnostic plots.

## Dataset

The UCF-Crime video files are **not** redistributed here. Obtain them from the original
authors under their licensing terms. The manifests reference paths of the form
`/kaggle/input/datasets/...`; adjust the roots to your own dataset location before running
the notebook.

## Reproducing training

`actionrec.ipynb` runs end to end on a GPU-enabled Kaggle session with both UCF-Crime
sources attached. Training on the deployment CPU was attempted and abandoned as impractically
slow; that machine is the inference target only.

## Citation

Cite the paper. The Zenodo all-versions DOI resolves to the latest release.

## License

Apache License 2.0. The dataset is governed separately by its original authors' terms.
