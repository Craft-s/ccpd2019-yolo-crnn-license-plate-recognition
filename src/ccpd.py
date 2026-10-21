from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
]

ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
PLATE_ALPHABET = "".join(PROVINCES) + "".join(ALPHABETS) + "".join(ADS)


@dataclass(frozen=True)
class CcpdAnnotation:
    image_path: Path
    bbox: tuple[int, int, int, int]
    points: list[tuple[int, int]]
    plate: str


def parse_ccpd_name(path: str | Path) -> CcpdAnnotation:
    image_path = Path(path)
    stem = image_path.stem
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Not a CCPD filename: {image_path.name}")

    bbox_raw = parts[2].split("_")
    left_top = tuple(map(int, bbox_raw[0].split("&")))
    right_bottom = tuple(map(int, bbox_raw[1].split("&")))
    x1, y1 = left_top
    x2, y2 = right_bottom

    points = []
    for pair in parts[3].split("_"):
        x, y = map(int, pair.split("&"))
        points.append((x, y))

    labels = list(map(int, parts[4].split("_")))
    if len(labels) != 7:
        raise ValueError(f"Unexpected plate label length in {image_path.name}")
    plate = PROVINCES[labels[0]] + ALPHABETS[labels[1]] + "".join(ADS[i] for i in labels[2:])

    return CcpdAnnotation(
        image_path=image_path,
        bbox=(x1, y1, x2, y2),
        points=points,
        plate=plate,
    )


def iter_ccpd_images(root: str | Path, subsets: Iterable[str] | None = None) -> list[Path]:
    root_path = Path(root)
    if subsets is None:
        subsets = [
            "ccpd_base",
            "ccpd_blur",
            "ccpd_challenge",
            "ccpd_db",
            "ccpd_fn",
            "ccpd_rotate",
            "ccpd_tilt",
            "ccpd_weather",
        ]

    images: list[Path] = []
    for subset in subsets:
        folder = root_path / subset
        if folder.exists():
            images.extend(sorted(folder.glob("*.jpg")))
    return images


def plate_charset() -> list[str]:
    seen: list[str] = []
    for ch in PLATE_ALPHABET:
        if ch not in seen:
            seen.append(ch)
    return seen
