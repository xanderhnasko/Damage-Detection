#!/usr/bin/env python3
"""Thin wrapper around Ultralytics YOLO training for building detection."""
import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/yolo_building.yaml", help="YOLO data yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--device", default="", help="CUDA device string, e.g., 0 or 0,1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    ap.add_argument("--project", default="~/project/outputs/yolo", help="Output project dir")
    ap.add_argument("--name", default="auto", help="Run name (auto => timestamped)")
    ap.add_argument("--exist_ok", action="store_true", help="Allow overwrite of existing run dir")
    args = ap.parse_args()

    project = str(Path(args.project).expanduser())
    name = args.name
    if name == "auto":
        name = datetime.now().strftime("yolo_%Y%m%d_%H%M%S")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project,
        name=name,
        exist_ok=args.exist_ok,
        lr0=args.lr0,
    )


if __name__ == "__main__":
    main()
