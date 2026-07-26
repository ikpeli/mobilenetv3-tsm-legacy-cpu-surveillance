# How core.py was produced

`core.py` is a mechanical extraction of the definitions in `actionrec.ipynb`, cells
13, 15, 17, 19, 21, 23, 25 and 27. It contains the frame reader, augmenter, dataset,
motion-score cache, class-balanced sampler, temporal shift module, model, focal loss,
metrics, checkpoint manager and epoch loop.

Three mechanical changes were applied during extraction, and nothing else:

1. Per-cell imports were consolidated into a single import block at the top.
2. Notebook driver statements (lines that built loaders, samplers or the optimizer at
   cell scope) were removed, because they run at import time and depend on names that
   only exist inside the notebook session. The function and class bodies are untouched.
3. Globals the notebook defined in surrounding cells (`tqdm`, `autocast`, `GradScaler`,
   `AMP_ENABLED`, `AMP_DEVICE_TYPE`, the shared `torch.Generator` `g`) are now defined
   at the top of the file, since the extracted code refers to them.

`CFG` is bound at runtime: `train.py` sets `core.CFG` before calling anything.

## Verification

Extraction was checked by building the model and comparing against the manuscript:

    python -c "import core; m=core.SuspiciousActivityModel(num_frames=8); \
      print(sum(p.numel() for p in m.parameters() if p.requires_grad))"
    # 2972913  -> matches the 2,972,913 reported in Section III-C

A forward pass on a random (2, 8, 3, 160, 160) batch returns the expected shape.

## If the notebook changes

Re-extract rather than editing `core.py` by hand. The notebook is the source of truth;
two hand-maintained copies of the same model will drift and the paper will end up
describing neither.
