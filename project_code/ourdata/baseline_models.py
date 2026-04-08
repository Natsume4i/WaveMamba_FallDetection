import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes: int = 2, in_chans: int = 1):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BiLSTMBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 232,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, 1, T, F], got shape {tuple(x.shape)}")
        x = x.squeeze(1)  # [B, T, F]
        out, _ = self.lstm(x)
        feat = out.mean(dim=1)
        logits = self.classifier(feat)
        return logits
