# Real-Time Suspicious-Activity Detection on Legacy CPU Hardware

This repository contains the reproducibility files for the MobileNetV3-Large + Temporal Shift Module (TSM) suspicious-activity detection study.

## Repository contents

- `actionrec.ipynb` — training, validation, testing, calibration, export and benchmarking notebook.
- `config.yaml` — dataset, split, training and evaluation configuration extracted from the notebook.
- `model_configuration.yaml` — architecture details for MobileNetV3-Large + TSM.
- `deployment_settings.yaml` — OpenVINO and alert-pipeline settings.
- `requirements.txt` — Python package requirements.
- `generate_manifests.py` — recreates the 500-video dataset manifest and the deterministic train/validation/test splits.
- `train_manifest.csv`
- `validation_manifest.csv`
- `test_manifest.csv`

## Important note about the three manifest CSV files

The uploaded notebook contains the code and random seed used to generate the splits, but it does **not embed the complete 500 file paths** in its saved cell outputs. Therefore, the three CSV files included here are schema-ready placeholders.

Run `generate_manifests.py` in the same Kaggle environment where the two UCF-Crime datasets are mounted. It will create the exact manifests using:

- all 100 Burglary videos;
- all 50 Fighting videos;
- all 100 Stealing videos;
- all 50 videos from `Normal_Videos_for_Event_Recognition`;
- 200 additional normal videos selected with seed 42 from `Training_Normal_Videos_Anomaly`;
- a 70:15:15 class-stratified, video-level split with seed 42.

## Expected split sizes

| Split | Total | Normal | Burglary | Fighting | Stealing |
|---|---:|---:|---:|---:|---:|
| Train | 350 | 175 | 70 | 35 | 70 |
| Validation | 75 | 38 | 15 | 7 | 15 |
| Test | 75 | 37 | 15 | 8 | 15 |

## Model summary

- Backbone: MobileNetV3-Large
- Temporal module: TSM before every inverted-residual block
- Frames per clip: 8
- Input size: 160 × 160
- Shift division: 8
- Dropout: 0.30
- Binary output: Normal vs Suspicious
- Trainable parameters reported by the manuscript: 2,972,913

## Main reported results

- Three-fold cross-validation PR-AUC: 0.870 ± 0.013
- Held-out video-level PR-AUC: 0.896
- Held-out video-level ROC-AUC: 0.902
- Event recall: 81.6%
- Event precision: 88.6%
- Event F1: 84.9%
- False-alarm rate: 10.8%
- OpenVINO FP32 latency in notebook benchmark: 38.0 ms per clip
- OpenVINO speed-up over PyTorch CPU: 1.59×

## Recreate the manifests on Kaggle

1. Attach these datasets:
   - `alirakhmaev/ucf-crime-full`
   - `vigneshwar472/ucaucf-crime-annotation-dataset`
2. Upload or clone this repository.
3. Run:

```bash
python generate_manifests.py
```

The script overwrites the three placeholder CSV files with the exact paths available in that Kaggle session.

## Data licence and redistribution

This repository should not redistribute UCF-Crime video files. It may publish only filenames, split manifests, configuration, code and derived model artefacts subject to the dataset and institutional policies.
