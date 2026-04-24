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
        self.classifier = nn.Linear(out_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, 1, T, F], got shape {tuple(x.shape)}")
        x = x.squeeze(1)  # [B, T, F]
        out, _ = self.lstm(x)  # [B, T, C]

        feat_mean = out.mean(dim=1)
        feat_max = out.max(dim=1).values
        feat = torch.cat([feat_mean, feat_max], dim=1)

        logits = self.classifier(feat)
        return logits


class GRUBaseline(nn.Module):
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
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, 1, T, F], got shape {tuple(x.shape)}")
        x = x.squeeze(1)  # [B, T, F]
        out, _ = self.gru(x)  # [B, T, C]

        feat_mean = out.mean(dim=1)
        feat_max = out.max(dim=1).values
        feat = torch.cat([feat_mean, feat_max], dim=1)

        logits = self.classifier(feat)
        return logits


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        if self.conv1.bias is not None:
            self.conv1.bias.data.zero_()
        if self.conv2.bias is not None:
            self.conv2.bias.data.zero_()
        if self.downsample is not None and self.downsample.bias is not None:
            self.downsample.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 232,
        num_channels=(64, 64, 128, 128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(num_channels[-1] * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, 1, T, F], got shape {tuple(x.shape)}")
        x = x.squeeze(1)  # [B, T, F]
        x = x.transpose(1, 2).contiguous()  # [B, F, T]
        feat_seq = self.tcn(x)  # [B, C, T]

        feat_mean = feat_seq.mean(dim=-1)
        feat_max = feat_seq.max(dim=-1).values
        feat = torch.cat([feat_mean, feat_max], dim=1)

        logits = self.classifier(feat)
        return logits


class VimBaseline(nn.Module):
    """
    Official ``VisionMamba`` from hustvl/Vim, code at ``<repo>/Vim/vim/`` (clone Vim.git next to this file).

    Input ``[B, 1, H, W]``; use Vim's ``mamba-1p1p1`` + ``causal-conv1d``, plus ``timm`` / ``einops``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        img_h: int = 500,
        img_w: int = 232,
        patch_size: int = 8,
        stride: int = 8,
        depth: int = 8,
        embed_dim: int = 128,
        d_state: int = 16,
        channels: int = 1,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        import sys
        from pathlib import Path

        vim_root = Path(__file__).resolve().parent / "Vim" / "vim"
        if not (vim_root / "models_mamba.py").is_file():
            raise FileNotFoundError(
                f"Vim code not found at {vim_root}. Clone https://github.com/hustvl/Vim.git "
                "into the project root so that Vim/vim/models_mamba.py exists."
            )
        root_str = str(vim_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        import models_mamba as vim_mamba

        rms_ok = vim_mamba.RMSNorm is not None and vim_mamba.rms_norm_fn is not None

        self.net = vim_mamba.VisionMamba(
            img_size=(img_h, img_w),
            patch_size=patch_size,
            stride=stride,
            depth=depth,
            embed_dim=embed_dim,
            d_state=d_state,
            channels=channels,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            rms_norm=rms_ok,
            fused_add_norm=rms_ok,
            residual_in_fp32=True,
            final_pool_type="mean",
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_divide_out=True,
            use_middle_cls_token=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, 1, H, W], got shape {tuple(x.shape)}")
        return self.net(x)