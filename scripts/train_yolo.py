from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector for CCPD license plates.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "project.yaml"))
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "yolo_plate" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_name = cfg.get("detector_model", "yolo26.pt")
    if not Path(model_name).exists() and model_name.endswith(".pt"):
        print(f"{model_name} not found, using {cfg.get('fallback_detector_model', 'yolo11n.pt')}")
        model_name = cfg.get("fallback_detector_model", "yolo11n.pt")

    model = YOLO(model_name)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_ROOT / "runs"),
        name="yolo_plate",
    )


if __name__ == "__main__":
    main()
