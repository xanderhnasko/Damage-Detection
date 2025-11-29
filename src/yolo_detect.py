#!/usr/bin/env python3
"""Run YOLO detection and emit a classifier-ready manifest CSV.
Optionally matches detections to GT manifest boxes to attach labels.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO


def load_gt_manifest(path: Path) -> Dict[str, List[dict]]:
    """Map image filename -> list of GT rows."""
    gt = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = Path(row["img_path"]).name
            gt[key].append(row)
    return gt


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="YOLO weights to use for inference")
    ap.add_argument("--data_root", default="data/yolo_buildings", help="Root of YOLO dataset (contains images/)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_csv", required=True, help="Where to write detection manifest")
    ap.add_argument("--manifest_gt", type=str, default=None, help="Optional GT manifest (for label lookup)")
    ap.add_argument("--match_iou", type=float, default=0.3, help="IoU threshold to assign GT label")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--nms_iou", type=float, default=0.45)
    ap.add_argument("--device", default="", help="CUDA device string, e.g., 0 or 0,1")
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--keep_unmatched", action="store_true", help="Include detections without GT match")
    args = ap.parse_args()

    source_dir = Path(args.data_root) / "images" / args.split
    if not source_dir.exists():
        raise SystemExit(f"Source split dir not found: {source_dir}")

    gt_map = {}
    if args.manifest_gt:
        gt_map = load_gt_manifest(Path(args.manifest_gt))
        print(f"[INFO] Loaded GT manifest with {sum(len(v) for v in gt_map.values())} rows")

    model = YOLO(args.weights)

    rows_out = []
    for res in model.predict(
        source=str(source_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        device=args.device,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    ):
        if res.boxes is None or res.boxes.xyxy is None:
            continue
        img_path = Path(res.path)
        key = img_path.name
        gt_rows = gt_map.get(key, [])

        for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            xmin, ymin, xmax, ymax = map(float, box)
            label_id, label_name, uid, pre_img_path = -1, "", "", ""
            best_iou = 0.0
            if gt_rows:
                for gr in gt_rows:
                    gbox = (
                        float(gr["xmin"]),
                        float(gr["ymin"]),
                        float(gr["xmax"]),
                        float(gr["ymax"]),
                    )
                    i = iou((xmin, ymin, xmax, ymax), gbox)
                    if i > best_iou:
                        best_iou = i
                        label_id = int(gr.get("label_id", -1))
                        label_name = gr.get("label_name", "")
                        uid = gr.get("uid", "")
                        pre_img_path = gr.get("pre_img_path", "")
            if label_id < 0 and not args.keep_unmatched:
                continue

            rows_out.append(
                {
                    "img_path": str(img_path),
                    "pre_img_path": pre_img_path,
                    "xmin": int(round(xmin)),
                    "ymin": int(round(ymin)),
                    "xmax": int(round(xmax)),
                    "ymax": int(round(ymax)),
                    "label_id": label_id,
                    "label_name": label_name,
                    "uid": uid,
                    "conf": float(conf),
                    "match_iou": best_iou,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "img_path",
                "pre_img_path",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "label_id",
                "label_name",
                "uid",
                "conf",
                "match_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] Wrote {len(rows_out)} detections to {out_csv}")


if __name__ == "__main__":
    main()
