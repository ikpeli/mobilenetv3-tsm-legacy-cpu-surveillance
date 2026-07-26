# README additions

Paste these two sections into the repository README. The first replaces the existing
"Repository contents" list; the second is new.

---

## Repository contents

- `actionrec.ipynb` — the master reproducibility notebook. Every result in the paper is
  produced here: training, threshold tuning, clip and event evaluation, calibration, CPU
  benchmarking, ONNX and OpenVINO export, INT8 evaluation, cross-validation and the data
  audits. See the map below.
- `core.py` — model, dataset, augmentation, loss, sampler, metrics, checkpoint manager
  and epoch loop, extracted from the notebook so they can be imported by a script.
- `train.py` — command-line training and evaluation entry point, for running the model
  without Jupyter.
- `EXTRACT_NOTES.md` — how `core.py` was extracted and how it was verified.
- `generate_manifests.py` — rebuilds the 500-video manifest and the seed-42 splits.
- `config.yaml`, `model_configuration.yaml`, `deployment_settings.yaml` — configuration.
- `requirements.txt` — pinned environment.
- `extracted_results.json` — the headline numbers reported in the paper.
- `train_manifest.csv`, `validation_manifest.csv`, `test_manifest.csv` — see the note on
  manifests below.
- `CITATION.cff` — citation metadata.

---

## Where each capability lives in the notebook

`train.py` covers training only. Everything else in the paper is in the notebook, at the
cell ranges below. Cell numbers refer to the 78-cell notebook as archived.

| Capability | Cells | Notebook sections |
|---|---|---|
| Setup and configuration | 0–7 | Environment install strategy; Core dependency setup; Dataset-path check; Global configuration |
| Dataset manifest and split | 8–11 | Expanded Normal manifest creation; Manifest loading and class-balanced split |
| Data pipeline | 12–17 | Video frame reader; Clip preprocessing and augmentation; DataLoader seeding and datasets |
| Model, loss, metrics, checkpoints | 18–27 | MobileNetV3-Large + TSM model; Focal loss; Metric computation; Checkpoint management; One training/evaluation epoch |
| Training | 28–29 | Main training loop |
| Threshold tuning and clip-level test evaluation | 30–35 | Tune clip threshold on validation; Evaluate the test set; Thresholded test metrics |
| Sliding-window inference and event-level alerting | 36–41 | Sliding-window video inference; Validation-tuned alert sweep |
| Calibration | 42–43 | Calibration analysis (ECE, reliability diagram) |
| CPU benchmarking | 44–47 | PyTorch CPU benchmark; trained-model benchmark with RAM tracking |
| ONNX and OpenVINO export | 48–61 | ONNX export and validation; OpenVINO FP32 conversion and benchmark |
| INT8 quantization | 62–65 | INT8 quantization; INT8 accuracy validation |
| Cross-validation | 66–69 | Evaluation loader helper; optional 3-fold cross-validation |
| Results export and audits | 70–77 | Summary export; manifest diagnostic; failure-case audit; data-source audit |

Two caveats worth stating plainly:

1. **The INT8 result in cell 62 was calibrated on 5 clips against a requested 300.** The
   measured PR-AUC drop is an upper bound obtained under poor calibration, not the best
   attainable INT8 accuracy. This is disclosed in the paper.
2. **The RTSP capture service and the alert dispatcher are not in this repository.** They
   ran on the deployment machine, not in the notebook. The alerting *logic* (EMA
   smoothing, N-consecutive confirmation, cooldown) is in cells 36–41 and is what
   produces the event-level numbers in the paper; the network service around it is not
   released.
