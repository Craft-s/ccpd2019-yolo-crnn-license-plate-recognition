from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ccpd import iter_ccpd_images, parse_ccpd_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def yolo_box(bbox: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "link":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO license plate detection data from CCPD2019.")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT.parent / "CCPD2019"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "yolo_plate"))
    parser.add_argument("--subsets", nargs="+", default=["ccpd_base", "ccpd_blur", "ccpd_rotate", "ccpd_tilt"])
    parser.add_argument("--max-images", type=int, default=4000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-mode", choices=["copy", "link"], default="copy")
    args = parser.parse_args()

    random.seed(args.seed)
    images = iter_ccpd_images(args.dataset_root, args.subsets)
    random.shuffle(images)
    if args.max_images > 0:
        images = images[: args.max_images]

    output = Path(args.output)
    split_at = max(1, int(len(images) * (1 - args.val_ratio)))
    splits = {"train": images[:split_at], "val": images[split_at:]}

    for split, split_images in splits.items():
        for image_path in tqdm(split_images, desc=f"YOLO {split}"):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            ann = parse_ccpd_name(image_path)
            dst_image = output / "images" / split / image_path.name
            dst_label = output / "labels" / split / f"{image_path.stem}.txt"
            copy_or_link(image_path, dst_image, args.file_mode)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text(yolo_box(ann.bbox, w, h) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(output).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "license_plate"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")
    print(f"YOLO data ready: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
