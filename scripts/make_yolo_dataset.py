#!/usr/bin/env python3
"""Build YOLO-format dataset from SpaceNet-style label JSONs.
Outputs images/{train,val,test} (symlinks by default) and labels/{split}/*.txt
with bbox annotations converted from polygons.
"""
import argparse, json, os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils_wkt import parse_wkt_polygon, bbox_from_points
from make_manifests import clip_bbox  # reuse bbox clipping logic


def load_events(config_path: str) -> Dict[str, List[str]]:
    """Load train/val/test event lists from a YAML config if provided."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "train": cfg.get("train_events", []) or [],
        "val": cfg.get("val_events", []) or [],
        "test": cfg.get("test_events", []) or [],
    }


def event_from_img_name(img_path: Path) -> str:
    """Infer event name from filename (prefix before first underscore)."""
    return img_path.name.split("_")[0]


def bbox_to_yolo(bb: List[int], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert xyxy bbox to YOLO-normalized xc, yc, w, h."""
    xmin, ymin, xmax, ymax = bb
    xc = (xmin + xmax) / (2.0 * width)
    yc = (ymin + ymax) / (2.0 * height)
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return xc, yc, w, h


def ensure_link(src: Path, dst: Path, copy: bool = False):
    """Create a symlink (or copy) for the image into the YOLO dataset tree."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        # Copy to avoid broken symlinks on remote filesystems
        import shutil
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def clear_label_dir(label_dir: Path):
    """Remove existing txt labels in a split dir to avoid stale files."""
    if not label_dir.exists():
        return
    for txt in label_dir.glob("*.txt"):
        txt.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, help="Root containing *_post_disaster images")
    ap.add_argument("--labels_root", required=True, help="Root containing label JSONs")
    ap.add_argument("--out_dir", required=True, help="Output dir for YOLO dataset (contains images/, labels/)")
    ap.add_argument("--config", type=str, default=None, help="YAML with train_events/val_events/test_events")
    ap.add_argument("--pad", type=int, default=8, help="Padding (pixels) around bbox before clipping")
    ap.add_argument("--val_frac", type=float, default=0.1, help="Val fraction when no event lists provided")
    ap.add_argument("--test_frac", type=float, default=0.0, help="Test fraction when no event lists provided")
    ap.add_argument("--copy_images", action="store_true", help="Copy images instead of symlinking")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    out_dir = Path(args.out_dir)

    # Resolve splits
    splits = {"train": [], "val": [], "test": []}
    event_splits = {"train": [], "val": [], "test": []}
    if args.config:
        event_splits = load_events(args.config)
        # If val_events not provided, fall back to test_events for validation to keep training workable
        if not event_splits["val"] and event_splits["test"]:
            event_splits["val"] = event_splits["test"]
        print(f"[INFO] Event-based split: {event_splits}")

    label_files = sorted(labels_root.rglob("*.json"))
    if not label_files:
        raise SystemExit(f"No label JSONs found under {labels_root}")

    # Prepare output dirs
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        clear_label_dir(out_dir / "labels" / split)

    # If no event-based split, pre-compute index thresholds
    n_labels = len(label_files)
    n_val = int(n_labels * args.val_frac) if not any(event_splits.values()) else 0
    n_test = int(n_labels * args.test_frac) if not any(event_splits.values()) else 0

    counts = defaultdict(int)
    skipped = 0

    for idx, lp in enumerate(tqdm(label_files, desc="Labels")):
        try:
            j = json.loads(lp.read_text())
        except Exception as e:
            print(f"[WARN] Failed to read {lp}: {e}")
            skipped += 1
            continue

        meta = j.get("metadata", {})
        img_name = meta.get("img_name") or (lp.stem + ".png")
        width = int(meta.get("width", meta.get("original_width", 0)))
        height = int(meta.get("height", meta.get("original_height", 0)))
        if width <= 0 or height <= 0:
            print(f"[WARN] Missing width/height in {lp}, skipping")
            skipped += 1
            continue

        img_path = images_root / img_name
        if not img_path.exists():
            print(f"[WARN] Image not found for {lp}, expected {img_path}")
            skipped += 1
            continue

        # Decide split
        ev = event_from_img_name(img_path)
        split = None
        if any(event_splits.values()):
            if ev in event_splits["train"]:
                split = "train"
            elif ev in event_splits["val"]:
                split = "val"
            elif ev in event_splits["test"]:
                split = "test"
        else:
            if idx < n_labels - (n_val + n_test):
                split = "train"
            elif idx < n_labels - n_test:
                split = "val"
            else:
                split = "test"

        if split is None:
            print(f"[WARN] Event {ev} not in split lists; skipping {lp}")
            skipped += 1
            continue

        xy = j.get("features", {}).get("xy", [])
        if not xy:
            continue

        label_lines = []
        for feat in xy:
            wkt = feat.get("wkt", "")
            if not wkt:
                continue
            pts = parse_wkt_polygon(wkt)
            bb = bbox_from_points(pts)
            bb = clip_bbox(bb, width, height, pad=args.pad)
            if not bb:
                continue
            xc, yc, w, h = bbox_to_yolo(bb, width, height)
            label_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not label_lines:
            continue

        # Write label file
        label_out = out_dir / "labels" / split / (img_path.stem + ".txt")
        label_out.write_text("\n".join(label_lines))

        # Link/copy image
        img_out = out_dir / "images" / split / img_path.name
        ensure_link(img_path.resolve(), img_out, copy=args.copy_images)

        counts[split] += 1

    print(f"[OK] YOLO dataset written to {out_dir}")
    print(f"Counts (by image JSON): {dict(counts)}, skipped: {skipped}")


if __name__ == "__main__":
    main()
