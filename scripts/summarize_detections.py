#!/usr/bin/env python3
"""Summarize detector CSV vs ground truth manifest."""
import argparse, csv
from pathlib import Path
from collections import Counter, defaultdict


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_csv", required=True, help="Detection CSV from yolo_detect.py")
    ap.add_argument("--manifest", required=True, help="GT manifest (for recall calc)")
    ap.add_argument("--conf_min", type=float, default=0.0, help="Min confidence to keep detections")
    ap.add_argument("--min_iou", type=float, default=0.0, help="Min match_iou to count as TP")
    args = ap.parse_args()

    det_rows = []
    with open(args.det_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["conf"] = float(r.get("conf", 0.0) or 0.0)
            r["match_iou"] = float(r.get("match_iou", 0.0) or 0.0)
            r["label_id"] = int(r.get("label_id", -1) or -1)
            det_rows.append(r)

    det_rows = [r for r in det_rows if r["conf"] >= args.conf_min]
    det_total = len(det_rows)
    tp_rows = [r for r in det_rows if r["label_id"] >= 0 and r["match_iou"] >= args.min_iou]
    matched = len(tp_rows)
    unmatched = det_total - matched

    # total GT boxes for images present in detections
    det_imgs = {Path(r["img_path"]).name for r in det_rows}
    gt_rows = load_manifest(args.manifest)
    gt_total = sum(1 for r in gt_rows if Path(r["img_path"]).name in det_imgs)

    precision = matched / det_total if det_total else 0.0
    recall = matched / gt_total if gt_total else 0.0

    print(f"Detections: {det_total}")
    print(f"Matched (TP): {matched}")
    print(f"Unmatched (FP): {unmatched}")
    print(f"GT boxes in evaluated images: {gt_total}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    # Simple per-conf bin breakdown
    bins = defaultdict(int)
    for r in det_rows:
        b = round(r["conf"], 1)
        bins[b] += 1
    if bins:
        print("Detections by conf (rounded):", dict(sorted(bins.items())))


if __name__ == "__main__":
    main()
