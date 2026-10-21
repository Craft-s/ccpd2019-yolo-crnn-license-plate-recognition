from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.crnn import CRNN, ctc_decode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def preprocess_plate(crop, width: int = 160, height: int = 32) -> torch.Tensor:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    img = resized.astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and recognize license plates in one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", default=str(PROJECT_ROOT / "runs" / "yolo_plate" / "weights" / "best.pt"))
    parser.add_argument("--recognizer", default=str(PROJECT_ROOT / "runs" / "crnn_plate_balanced" / "best.pt"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "prediction.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    detector = YOLO(args.detector)
    checkpoint = torch.load(args.recognizer, map_location=args.device)
    charset = checkpoint["charset"]
    recognizer = CRNN(num_classes=len(charset) + 1).to(args.device)
    recognizer.load_state_dict(checkpoint["model"])
    recognizer.eval()

    results = detector.predict(source=img, conf=args.conf, verbose=False)
    predictions = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        with torch.no_grad():
            log_probs = recognizer(preprocess_plate(crop).to(args.device))
            text = ctc_decode(log_probs, charset)[0]
        predictions.append({"box": [x1, y1, x2, y2], "text": text, "score": float(box.conf[0])})
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)
    for item in predictions:
        print(item)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
