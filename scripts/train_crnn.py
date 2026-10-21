from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.crnn import CRNN, ctc_decode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PlateDataset(Dataset):
    def __init__(self, root: str | Path, split: str, charset: list[str]):
        self.root = Path(root)
        self.split = split
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(charset)}
        label_file = self.root / f"{split}.txt"
        self.samples = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, text = line.split("\t", 1)
            if all(ch in self.char_to_idx for ch in text):
                self.samples.append((name, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, text = self.samples[idx]
        img = cv2.imread(str(self.root / self.split / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(name)
        img = img.astype("float32") / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).unsqueeze(0)
        target = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        return tensor, target, text


def collate(batch):
    images, targets, texts = zip(*batch)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(images), torch.cat(targets), target_lengths, list(texts)


@torch.no_grad()
def validate(model: CRNN, loader: DataLoader, charset: list[str], device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for images, _, _, texts in loader:
        images = images.to(device)
        preds = ctc_decode(model(images), charset)
        total += len(texts)
        correct += sum(p == t for p, t in zip(preds, texts))
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRNN recognizer for CCPD plates.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "crnn_plate_balanced"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "crnn_plate_balanced" / "best.pt"))
    args = parser.parse_args()

    data_root = Path(args.data)
    charset = json.loads((data_root / "charset.json").read_text(encoding="utf-8"))
    train_set = PlateDataset(data_root, "train", charset)
    val_set = PlateDataset(data_root, "val", charset)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    device = torch.device(args.device)
    model = CRNN(num_classes=len(charset) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, target_lengths, _ in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        acc = validate(model, val_loader, charset, device)
        avg_loss = sum(losses) / max(1, len(losses))
        print(f"epoch={epoch} loss={avg_loss:.4f} val_exact_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "charset": charset, "val_exact_acc": acc}, output)
            print(f"saved: {output}")


if __name__ == "__main__":
    main()
