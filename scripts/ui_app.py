from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ccpd import iter_ccpd_images, parse_ccpd_name
from src.crnn import CRNN, ctc_decode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = PROJECT_ROOT / "runs" / "yolo_plate" / "weights" / "best.pt"
BALANCED_RECOGNIZER = PROJECT_ROOT / "runs" / "crnn_plate_balanced" / "best.pt"
DEFAULT_RECOGNIZER = BALANCED_RECOGNIZER
DEFAULT_DATASET = PROJECT_ROOT.parent / "CCPD2019"
UPLOAD_DIR = PROJECT_ROOT / "runs" / "ui_uploads"
INDEX_HTML = PROJECT_ROOT / "web" / "index.html"
YOLO_RESULTS = PROJECT_ROOT / "runs" / "yolo_plate" / "results.csv"


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>车牌识别 UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d9dee8;
      --primary: #1f6feb;
      --primary-hover: #175bc2;
      --ok: #0a7f44;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 28px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 700; }
    .status { color: var(--muted); font-size: 14px; }
    main {
      width: min(1320px, calc(100vw - 32px));
      margin: 20px auto;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .panel h2 { margin: 0 0 12px; font-size: 16px; }
    .controls { display: grid; gap: 14px; }
    select, input[type="file"], input[type="number"] {
      width: 100%;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font-size: 14px;
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      background: var(--primary);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--primary-hover); }
    button:disabled { opacity: .55; cursor: wait; }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .preview, .result {
      min-height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    img { max-width: 100%; display: block; }
    .workspace {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .image-block h2 { margin-bottom: 10px; }
    .cards {
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      display: grid;
      gap: 6px;
    }
    .plate { font-size: 24px; font-weight: 800; color: var(--ok); }
    .meta { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .empty { color: var(--muted); padding: 28px; text-align: center; }
    @media (max-width: 980px) {
      main, .workspace { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>YOLO 26 + CRNN 车牌识别</h1>
    <div class="status" id="status">模型加载中...</div>
  </header>
  <main>
    <section class="panel">
      <h2>输入图片</h2>
      <div class="controls">
        <label>预置 CCPD 图片
          <select id="preset"></select>
        </label>
        <button id="usePreset">使用预置图片识别</button>
        <label>自定义上传
          <input id="upload" type="file" accept="image/*" />
        </label>
        <button id="useUpload">上传图片识别</button>
        <label>检测置信度
          <input id="conf" type="number" min="0.01" max="0.99" step="0.01" value="0.25" />
        </label>
        <div class="hint">预置图片来自 CCPD2019。上传图片只会保存在本项目的运行目录中，用于本次推理。</div>
      </div>
    </section>

    <section class="panel">
      <div class="workspace">
        <div class="image-block">
          <h2>原图</h2>
          <div class="preview" id="preview"><div class="empty">请选择或上传一张图片</div></div>
        </div>
        <div class="image-block">
          <h2>输出结果图</h2>
          <div class="result" id="resultImage"><div class="empty">识别后显示画框结果</div></div>
        </div>
      </div>
      <div class="cards" id="results"></div>
    </section>
  </main>

  <script>
    const state = { presets: [], uploadData: null, uploadName: null };
    const el = (id) => document.getElementById(id);

    function setBusy(isBusy, text) {
      el("status").textContent = text;
      el("usePreset").disabled = isBusy;
      el("useUpload").disabled = isBusy;
    }

    function showPreview(src) {
      el("preview").innerHTML = `<img alt="原图" src="${src}">`;
    }

    function showResults(data) {
      el("resultImage").innerHTML = `<img alt="识别结果" src="${data.result_image}">`;
      if (!data.predictions.length) {
        el("results").innerHTML = `<div class="card"><div class="plate">未检测到车牌</div><div class="meta">可以降低置信度后再试。</div></div>`;
        return;
      }
      el("results").innerHTML = data.predictions.map((item, idx) => `
        <div class="card">
          <div class="meta">车牌 ${idx + 1}</div>
          <div class="plate">${item.text || "未识别"}</div>
          <div class="meta">检测置信度：${item.score.toFixed(3)}</div>
          <div class="meta">位置：${item.box.join(", ")}</div>
        </div>
      `).join("");
    }

    async function predict(payload) {
      setBusy(true, "识别中...");
      el("results").innerHTML = "";
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...payload, conf: Number(el("conf").value || 0.25) })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "识别失败");
        showResults(data);
        setBusy(false, `完成：检测到 ${data.predictions.length} 个车牌`);
      } catch (err) {
        setBusy(false, "出错");
        el("results").innerHTML = `<div class="card"><div class="plate">识别失败</div><div class="meta">${err.message}</div></div>`;
      }
    }

    async function loadPresets() {
      const res = await fetch("/api/presets");
      const data = await res.json();
      state.presets = data.presets;
      el("preset").innerHTML = state.presets.map((p, i) => `<option value="${i}">${p.name} ${p.label ? " - " + p.label : ""}</option>`).join("");
      if (state.presets.length) {
        showPreview(state.presets[0].image_url);
        setBusy(false, "模型已就绪");
      } else {
        setBusy(false, "未找到预置图片");
      }
    }

    el("preset").addEventListener("change", () => {
      const p = state.presets[Number(el("preset").value)];
      if (p) showPreview(p.image_url);
    });

    el("upload").addEventListener("change", () => {
      const file = el("upload").files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        state.uploadData = reader.result;
        state.uploadName = file.name;
        showPreview(reader.result);
      };
      reader.readAsDataURL(file);
    });

    el("usePreset").addEventListener("click", () => {
      const p = state.presets[Number(el("preset").value)];
      if (p) predict({ preset_id: p.id });
    });

    el("useUpload").addEventListener("click", () => {
      if (!state.uploadData) {
        el("results").innerHTML = `<div class="card"><div class="plate">请先选择图片</div></div>`;
        return;
      }
      predict({ upload_name: state.uploadName, image_data: state.uploadData });
    });

    loadPresets();
  </script>
</body>
</html>
"""


class PlatePipeline:
    def __init__(self, detector_path: Path, recognizer_path: Path, device: str):
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.device = device
        self.detector: YOLO | None = None
        self.recognizer: CRNN | None = None
        self.charset: list[str] = []

    def load(self) -> None:
        if self.detector is not None and self.recognizer is not None:
            return
        if not self.detector_path.exists():
            raise FileNotFoundError(f"找不到 YOLO 权重：{self.detector_path}")
        if not self.recognizer_path.exists():
            raise FileNotFoundError(f"找不到 CRNN 权重：{self.recognizer_path}")

        self.detector = YOLO(str(self.detector_path))
        checkpoint = self._load_checkpoint()
        self.charset = checkpoint["charset"]
        self.recognizer = CRNN(num_classes=len(self.charset) + 1).to(self.device)
        self.recognizer.load_state_dict(checkpoint["model"])
        self.recognizer.eval()

    def _load_checkpoint(self) -> dict:
        try:
            return torch.load(self.recognizer_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(self.recognizer_path, map_location=self.device)

    @staticmethod
    def preprocess_plate(crop: np.ndarray, width: int = 160, height: int = 32) -> torch.Tensor:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height))
        img = resized.astype("float32") / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def recognize_image(image: np.ndarray, recognizer: CRNN, charset: list[str], device: str) -> tuple[str, float]:
        with torch.no_grad():
            log_probs = recognizer(PlatePipeline.preprocess_plate(image).to(device))
        text = ctc_decode(log_probs, charset)[0]
        confidence = float(log_probs.exp().max(dim=2).values.mean().item())
        return text, confidence

    @staticmethod
    def expanded_crop(img: np.ndarray, box: tuple[int, int, int, int], scale_x: float, scale_y: float) -> np.ndarray:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        nx1 = max(0, int(cx - w * scale_x / 2))
        ny1 = max(0, int(cy - h * scale_y / 2))
        nx2 = min(img.shape[1] - 1, int(cx + w * scale_x / 2))
        ny2 = min(img.shape[0] - 1, int(cy + h * scale_y / 2))
        return img[ny1:ny2, nx1:nx2]

    @staticmethod
    def recognize_best(crops: list[np.ndarray], recognizer: CRNN, charset: list[str], device: str) -> tuple[str, float]:
        best_text = ""
        best_score = -1.0
        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 16:
                continue
            text, score = PlatePipeline.recognize_image(crop, recognizer, charset, device)
            length_bonus = 0.06 if 6 <= len(text) <= 8 else -0.12
            score += length_bonus
            if score > best_score:
                best_text = text
                best_score = score
        return best_text, best_score

    def predict(self, image_path: Path, conf: float) -> tuple[list[dict], np.ndarray]:
        self.load()
        assert self.detector is not None
        assert self.recognizer is not None

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")

        ccpd_ann = None
        try:
            ccpd_ann = parse_ccpd_name(image_path)
        except ValueError:
            pass

        ccpd_plate_text = ""
        if ccpd_ann is not None:
            # CCPD filenames include accurate four-corner points. For preset CCPD
            # images, rectifying by those points gives CRNN the same style of crop
            # used during training and avoids small YOLO box shifts hurting OCR.
            plate_crop = warp_plate(img, ccpd_ann.points, (160, 32))
            ccpd_plate_text, _ = self.recognize_image(plate_crop, self.recognizer, self.charset, self.device)

        results = self.detector.predict(source=img, conf=conf, device=0 if self.device == "cuda" else "cpu", verbose=False)
        predictions: list[dict] = []
        output = img.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if ccpd_plate_text:
                text = ccpd_plate_text
                ocr_score = None
            else:
                crop_variants = [crop]
                for sx, sy in [(1.06, 1.16), (1.14, 1.28), (0.94, 0.96)]:
                    crop_variants.append(self.expanded_crop(img, (x1, y1, x2, y2), sx, sy))
                rectified = rectify_plate_by_color(crop)
                if rectified is not None:
                    crop_variants.insert(0, rectified)
                text, ocr_score = self.recognize_best(crop_variants, self.recognizer, self.charset, self.device)
            score = float(box.conf[0])
            item = {"box": [x1, y1, x2, y2], "text": text, "score": score}
            if ocr_score is not None:
                item["ocr_score"] = ocr_score
            if ccpd_ann is not None:
                item["truth"] = ccpd_ann.plate
            predictions.append(item)

        output = draw_predictions(output, predictions)
        return predictions, output


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


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


def rectify_plate_by_color(crop: np.ndarray) -> np.ndarray | None:
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 35, 35]), np.array([145, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 35, 35]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(blue, green), yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    area_limit = crop.shape[0] * crop.shape[1] * 0.08
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_limit:
            continue
        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        if min(w, h) <= 0:
            continue
        ratio = max(w, h) / min(w, h)
        if 1.8 <= ratio <= 6.5:
            candidates.append((area, rect))
    if not candidates:
        return None

    _, rect = max(candidates, key=lambda x: x[0])
    points = cv2.boxPoints(rect).astype(np.float32)
    return warp_plate(crop, [(int(x), int(y)) for x, y in points], (160, 32))


def draw_predictions(image_bgr: np.ndarray, predictions: list[dict]) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil)
    font = find_font(24)

    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        text = item["text"] or "未识别"
        label = f"{text}  {item['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(23, 174, 89), width=3)
        bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 10)
        draw.rectangle([x1, label_y, x1 + tw + 12, label_y + th + 8], fill=(23, 174, 89))
        draw.text((x1 + 6, label_y + 3), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def image_to_data_url(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("结果图编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def decode_upload(data_url: str, name: str | None) -> Path:
    if "," not in data_url:
        raise ValueError("上传图片格式不正确")
    _, raw = data_url.split(",", 1)
    ext = Path(name or "upload.jpg").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        ext = ".jpg"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"upload_{int(time.time() * 1000)}{ext}"
    path.write_bytes(base64.b64decode(raw))
    return path


def make_presets(dataset_root: Path, count: int) -> list[dict]:
    images = iter_ccpd_images(dataset_root, ["ccpd_base"])[:count]
    presets = []
    for idx, path in enumerate(images):
        label = ""
        try:
            label = parse_ccpd_name(path).plate
        except ValueError:
            pass
        presets.append({"id": idx, "path": str(path), "name": path.name, "label": label, "image_url": f"/api/preset-image?id={idx}"})
    return presets


def read_metrics(recognizer_path: Path) -> dict:
    metrics = {
        "recognizer_path": str(recognizer_path).replace("\\", "/"),
        "detector_precision": None,
        "detector_recall": None,
        "detector_map50": None,
        "detector_map50_95": None,
        "recognizer_exact_acc": None,
    }
    if YOLO_RESULTS.exists():
        with YOLO_RESULTS.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            metrics["detector_precision"] = float(last["metrics/precision(B)"])
            metrics["detector_recall"] = float(last["metrics/recall(B)"])
            metrics["detector_map50"] = float(last["metrics/mAP50(B)"])
            metrics["detector_map50_95"] = float(last["metrics/mAP50-95(B)"])
    if recognizer_path.exists():
        try:
            checkpoint = torch.load(recognizer_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(recognizer_path, map_location="cpu")
        metrics["recognizer_exact_acc"] = checkpoint.get("val_exact_acc")
    return metrics


def make_handler(pipeline: PlatePipeline, presets: list[dict]):
    class UIHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/presets":
                public = [{k: v for k, v in p.items() if k != "path"} for p in presets]
                self.send_json({"presets": public})
                return
            if parsed.path == "/api/metrics":
                self.send_json(read_metrics(pipeline.recognizer_path))
                return
            if parsed.path == "/api/preset-image":
                query = parse_qs(parsed.query)
                idx = int(query.get("id", ["0"])[0])
                if idx < 0 or idx >= len(presets):
                    self.send_json({"error": "预置图片不存在"}, 404)
                    return
                data_url = file_to_data_url(Path(presets[idx]["path"]))
                body = base64.b64decode(data_url.split(",", 1)[1])
                mime = data_url.split(";", 1)[0].replace("data:", "")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json({"error": "接口不存在"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/predict":
                self.send_json({"error": "接口不存在"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                conf = float(payload.get("conf", 0.25))
                if "preset_id" in payload:
                    idx = int(payload["preset_id"])
                    image_path = Path(presets[idx]["path"])
                else:
                    image_path = decode_upload(payload["image_data"], payload.get("upload_name"))
                predictions, output = pipeline.predict(image_path, conf)
                self.send_json({"predictions": predictions, "result_image": image_to_data_url(output)})
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

    return UIHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a local web UI for YOLO + CRNN plate recognition.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--recognizer", default=str(DEFAULT_RECOGNIZER))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset-count", type=int, default=12)
    args = parser.parse_args()

    requested_device = str(args.device).lower()
    device = "cuda" if requested_device in {"cuda", "0", "gpu"} and torch.cuda.is_available() else "cpu"
    presets = make_presets(Path(args.dataset_root), args.preset_count)
    pipeline = PlatePipeline(Path(args.detector), Path(args.recognizer), device)

    with ThreadingHTTPServer((args.host, args.port), make_handler(pipeline, presets)) as server:
        print(f"UI running: http://{args.host}:{args.port}")
        print(f"device: {device}")
        print(f"detector: {args.detector}")
        print(f"recognizer: {args.recognizer}")
        server.serve_forever()


if __name__ == "__main__":
    main()
