from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

CURRENT_ROOT = Path("/kaggle/input/datasets/alirakhmaev/ucf-crime-full")
FULL_VIDEOS_ROOT = Path(
    "/kaggle/input/datasets/vigneshwar472/"
    "ucaucf-crime-annotation-dataset/UCF_Crimes/UCF_Crimes/Videos"
)

OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_NORMAL_TOTAL = 250


def list_videos(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def build_master_manifest():
    records = []

    for class_name in ("Burglary", "Fighting", "Stealing"):
        videos = list_videos(CURRENT_ROOT / class_name)
        for path in videos:
            records.append({
                "video_path": str(path),
                "class_name": class_name,
                "label": 1,
            })

    current_normals = list_videos(
        CURRENT_ROOT / "Normal_Videos_for_Event_Recognition"
    )
    for path in current_normals:
        records.append({
            "video_path": str(path),
            "class_name": "Normal",
            "label": 0,
        })

    full_normals = list_videos(
        FULL_VIDEOS_ROOT / "Training_Normal_Videos_Anomaly"
    )

    current_names = {p.name for p in current_normals}
    extra_normals = [p for p in full_normals if p.name not in current_names]

    rng = random.Random(SEED)
    rng.shuffle(extra_normals)

    needed = TARGET_NORMAL_TOTAL - len(current_normals)
    selected = extra_normals[:needed]

    if len(selected) != needed:
        raise RuntimeError(
            f"Needed {needed} extra normal videos, found only {len(selected)}."
        )

    for path in selected:
        records.append({
            "video_path": str(path),
            "class_name": "Normal",
            "label": 0,
        })

    manifest = pd.DataFrame(records)

    expected = {
        "Normal": 250,
        "Burglary": 100,
        "Fighting": 50,
        "Stealing": 100,
    }
    actual = manifest["class_name"].value_counts().to_dict()
    if actual != expected:
        raise RuntimeError(f"Unexpected class counts: {actual}; expected {expected}")

    if manifest["video_path"].str.contains(
        "Testing_Normal_Videos_Anomaly", regex=False
    ).any():
        raise RuntimeError("Testing-normal leakage detected.")

    return manifest


def split_manifest(manifest):
    records = manifest.to_dict("records")
    labels = [r["class_name"] for r in records]

    train, remainder = train_test_split(
        records,
        train_size=0.70,
        stratify=labels,
        random_state=SEED,
    )

    remainder_labels = [r["class_name"] for r in remainder]
    validation, test = train_test_split(
        remainder,
        train_size=0.50,
        stratify=remainder_labels,
        random_state=SEED,
    )

    train_paths = {r["video_path"] for r in train}
    val_paths = {r["video_path"] for r in validation}
    test_paths = {r["video_path"] for r in test}

    if train_paths & val_paths or train_paths & test_paths or val_paths & test_paths:
        raise RuntimeError("Video leakage detected across splits.")

    return (
        pd.DataFrame(train),
        pd.DataFrame(validation),
        pd.DataFrame(test),
    )


def main():
    manifest = build_master_manifest()
    train, validation, test = split_manifest(manifest)

    manifest.to_csv(OUTPUT_DIR / "expanded_normal_manifest.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train_manifest.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_manifest.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_manifest.csv", index=False)

    print("Created manifests:")
    for name, df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        print(f"\n{name}: {len(df)} videos")
        print(df["class_name"].value_counts().sort_index())
        print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
