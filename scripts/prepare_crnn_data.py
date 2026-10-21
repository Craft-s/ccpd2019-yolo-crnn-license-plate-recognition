from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ccpd import iter_ccpd_images, parse_ccpd_name, plate_charset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ordered_quad(points: list[tuple[int, int]]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def warp_plate(image: np.ndarray, points: list[tuple[int, int]], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    src = ordered_quad(points)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop CCPD plates for CRNN recognition.")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT.parent / "CCPD2019"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "crnn_plate_balanced"))
    parser.add_argument("--subsets", nargs="+", default=["ccpd_base", "ccpd_blur", "ccpd_rotate", "ccpd_tilt"])
    parser.add_argument("--max-images", type=int, default=4000)
    parser.add_argument("--balanced", action="store_true", default=True, help="Sample images per province instead of plain random sampling.")
    parser.add_argument("--max-per-province", type=int, default=800, help="Used only with --balanced.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    images = iter_ccpd_images(args.dataset_root, args.subsets)
    if args.balanced:
        buckets = defaultdict(list)
        for image_path in images:
            try:
                province = parse_ccpd_name(image_path).plate[0]
            except ValueError:
                continue
            buckets[province].append(image_path)
        images = []
        for province_images in buckets.values():
            random.shuffle(province_images)
            images.extend(province_images[: args.max_per_province])
        random.shuffle(images)
    else:
        random.shuffle(images)
        if args.max_images > 0:
            images = images[: args.max_images]

    output = Path(args.output)
    split_at = max(1, int(len(images) * (1 - args.val_ratio)))
    splits = {"train": images[:split_at], "val": images[split_at:]}

    for split, split_images in splits.items():
        lines: list[str] = []
        image_dir = output / split
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, image_path in enumerate(tqdm(split_images, desc=f"CRNN {split}")):
            ann = parse_ccpd_name(image_path)
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            crop = warp_plate(img, ann.points, (args.width, args.height))
            crop_name = f"{split}_{idx:06d}.jpg"
            cv2.imwrite(str(image_dir / crop_name), crop)
            lines.append(f"{crop_name}\t{ann.plate}")
        (output / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    charset_file = output / "charset.json"
    charset_file.write_text(json.dumps(plate_charset(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CRNN data ready: {output}")


if __name__ == "__main__":
    main()
