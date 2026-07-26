#!/usr/bin/env python3
"""
train.py - command-line training entry point for the MobileNetV3-Large + TSM
suspicious-activity detector.

This is a thin, runnable wrapper around the code in actionrec.ipynb. It exists so the
experiment can be reproduced without Jupyter. The model, dataset, augmentation, loss,
sampler and training-loop definitions are imported from core.py, which is extracted
verbatim from the notebook; nothing is reimplemented here, so the two cannot drift.

Typical use:

    # 1. build the manifests (deterministic, seed 42)
    python generate_manifests.py

    # 2. train
    python train.py --data-root /path/to/UCF_Crimes \
                    --train-manifest train_manifest.csv \
                    --val-manifest validation_manifest.csv \
                    --out checkpoints/

    # 3. evaluate the retained checkpoint on the held-out partition
    python train.py --data-root /path/to/UCF_Crimes \
                    --test-manifest test_manifest.csv \
                    --resume checkpoints/best_prauc.pt --eval-only

Reference run: stopped by early stopping at epoch 21 on an NVIDIA T4, roughly 282 s per
epoch, with the epoch-13 checkpoint retained (validation PR-AUC 0.846, ROC-AUC 0.863).
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import core
from core import (
    UCFCrimeClipDataset, SuspiciousActivityModel, FocalLoss, CheckpointManager,
    build_class_balanced_sampler, make_eval_loader, compute_metrics, run_epoch,
    seed_worker, set_seed, g, GradScaler, AMP_ENABLED,
)


class Config:
    """Mirrors the CFG class in the notebook. Values are the reference-run defaults."""
    IMG_SIZE = 160
    NUM_FRAMES = 8
    INFER_STRIDE = 4
    TRAIN_SAMPLES_PER_VIDEO = 2

    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 8
    NUM_WORKERS = 4

    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.5
    HARD_NEGATIVE_TOPK_FRAC = 0.15

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    SEED = 42
    OUTPUT_DIR = "checkpoints"
    MOTION_CACHE_PATH = "motion_cache.json"

    SAVE_BEST_PRAUC = "best_prauc.pt"
    SAVE_BEST_ROCAUC = "best_rocauc.pt"
    SAVE_LATEST = "latest.pt"


def records_from_manifest(csv_path, data_root):
    """Manifest CSV -> the record dicts the notebook's dataset class expects."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(
            f"{csv_path} contains no rows. The manifests shipped in this repository are "
            f"schema stubs; run generate_manifests.py first."
        )
    records = []
    for _, r in df.iterrows():
        p = str(r["video_path"])
        if data_root and not os.path.isabs(p):
            p = os.path.join(data_root, p)
        records.append({"path": p, "class_name": r.get("class_name"),
                        "label": int(r["label"])})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="", help="prefix for relative manifest paths")
    ap.add_argument("--train-manifest", default="train_manifest.csv")
    ap.add_argument("--val-manifest", default="validation_manifest.csv")
    ap.add_argument("--test-manifest", default="")
    ap.add_argument("--out", default=Config.OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=Config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=Config.LR)
    ap.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS)
    ap.add_argument("--resume", default="", help="checkpoint to resume or evaluate")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=Config.SEED)
    a = ap.parse_args()

    cfg = Config()
    cfg.OUTPUT_DIR, cfg.EPOCHS = a.out, a.epochs
    cfg.BATCH_SIZE, cfg.LR, cfg.NUM_WORKERS = a.batch_size, a.lr, a.num_workers
    cfg.SEED = a.seed
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    set_seed(cfg.SEED)

    core.CFG = cfg          # the extracted notebook code reads a module-level CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SuspiciousActivityModel(num_frames=cfg.NUM_FRAMES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")   # reference run: 2,972,913

    if a.resume:
        state = torch.load(a.resume, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        print(f"loaded checkpoint: {a.resume}")

    if a.eval_only:
        if not a.test_manifest:
            raise SystemExit("--eval-only requires --test-manifest")
        ds = UCFCrimeClipDataset(records_from_manifest(a.test_manifest, a.data_root),
                                 cfg, train=False)
        loader = make_eval_loader(ds, cfg)
        model.eval()
        probs, targets = [], []
        with torch.inference_mode():
            for clips, labels, _paths in loader:
                logits = model(clips.to(device))
                probs += torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                targets += labels.tolist()
        metrics = compute_metrics(np.array(targets), np.array(probs))
        print(json.dumps(metrics, indent=2, default=float))
        return

    train_records = records_from_manifest(a.train_manifest, a.data_root)
    val_records = records_from_manifest(a.val_manifest, a.data_root)
    print(f"train videos: {len(train_records)}   validation videos: {len(val_records)}")

    train_ds = UCFCrimeClipDataset(train_records, cfg, train=True)
    val_ds = UCFCrimeClipDataset(val_records, cfg, train=False)
    # returns (sampler, motion_cache); the cache is persisted so MOG2 motion scores
    # are computed once and reused across runs
    sampler, motion_cache = build_class_balanced_sampler(
        train_records, cfg, hard_negative_topk_frac=cfg.HARD_NEGATIVE_TOPK_FRAC,
        motion_cache=core.load_motion_cache(cfg.MOTION_CACHE_PATH))
    core.save_motion_cache(motion_cache, cfg.MOTION_CACHE_PATH)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
        num_workers=cfg.NUM_WORKERS, worker_init_fn=seed_worker, generator=g,
        pin_memory=True, drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = make_eval_loader(val_ds, cfg)

    criterion = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)
    scaler = GradScaler(enabled=AMP_ENABLED)
    ckpt = CheckpointManager(cfg.OUTPUT_DIR, cfg)

    best, stale = -1.0, 0
    for epoch in range(1, cfg.EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, device, scaler,
                       train=True, desc=f"epoch {epoch} train")
        va = run_epoch(model, val_loader, optimizer, criterion, device, scaler,
                       train=False, desc=f"epoch {epoch} val")
        pr_auc = va.get("pr_auc", 0.0)
        scheduler.step(pr_auc)
        print(f"epoch {epoch:02d}  train_loss {tr.get('loss', 0):.4f}  "
              f"val_pr_auc {pr_auc:.4f}  val_roc_auc {va.get('roc_auc', 0):.4f}")
        ckpt.update(model, optimizer, epoch, va, scheduler=scheduler, scaler=scaler)
        if pr_auc > best:
            best, stale = pr_auc, 0
        else:
            stale += 1
            if stale >= cfg.EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch}; best val PR-AUC {best:.4f}")
                break

    print(f"done. best validation PR-AUC {best:.4f}. checkpoints in {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
