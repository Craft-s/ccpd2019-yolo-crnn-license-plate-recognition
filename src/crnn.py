from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(512 * 2, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        b, c, h, w = feats.shape
        feats = feats.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        seq, _ = self.rnn(feats)
        return self.classifier(seq).log_softmax(2)


def ctc_decode(log_probs: torch.Tensor, charset: list[str]) -> list[str]:
    preds = log_probs.argmax(dim=2)
    results: list[str] = []
    for row in preds:
        text = []
        prev = -1
        for idx in row.tolist():
            if idx != 0 and idx != prev:
                text.append(charset[idx - 1])
            prev = idx
        results.append("".join(text))
    return results
