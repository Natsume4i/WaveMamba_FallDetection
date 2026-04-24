import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# Try importing official Mamba
# ==========================================================
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("[Info] Official Mamba-SSM library loaded successfully.")
except ImportError:
    HAS_MAMBA = False
    print("[Warning] Mamba-SSM not found. Falling back to linear placeholder.")


# ==========================================================
# 1. Basic modules
# ==========================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with safe hidden width."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class HaarWaveletTransform1D(nn.Module):
    """
    Fixed 1D Haar wavelet decomposition along the temporal dimension.

    Input : [B, C, T, F]
    Output: [B, C, 2, T/2, F]
    """
    def __init__(self, in_channels: int):
        super().__init__()
        dec_lo = torch.tensor([0.70710678, 0.70710678], dtype=torch.float32)
        dec_hi = torch.tensor([-0.70710678, 0.70710678], dtype=torch.float32)

        l_filter = dec_lo.view(2, 1)
        h_filter = dec_hi.view(2, 1)

        filters = torch.stack([l_filter, h_filter], dim=0)
        filters = filters.unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        if t % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        out = F.conv2d(x, self.filters, stride=(2, 1), groups=c)
        return out.view(b, c, 2, out.shape[2], out.shape[3])


class BiMambaBlock(nn.Module):
    """
    Lightweight bidirectional Mamba block.

    It applies LayerNorm, processes the sequence in both forward and
    reversed directions with two independent Mamba modules, fuses the
    two outputs by a linear projection, and adds a residual connection.

    Input / Output: [B, L, C]
    """
    def __init__(self, dim: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        if HAS_MAMBA:
            self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=expand)
            self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=expand)
            self.out_proj = nn.Linear(dim, dim)
        else:
            self.linear_fallback = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)

        if HAS_MAMBA:
            out_fwd = self.mamba_fwd(x_norm)
            out_bwd = self.mamba_bwd(x_norm.flip([1])).flip([1])
            out = self.out_proj(out_fwd + out_bwd)
        else:
            out = self.linear_fallback(x_norm)

        return out + residual


class MultiKernelConv(nn.Module):
    """
    Multi-kernel depthwise convolution branch.

    The channels are split into two parts:
      - first part: 3x3 depthwise convolution
      - second part: 5x5 depthwise convolution
    """
    def __init__(self, channels: int):
        super().__init__()
        if channels < 2:
            raise ValueError(f"MultiKernelConv requires channels >= 2, got {channels}")

        self.c1 = channels // 2
        self.c2 = channels - self.c1

        self.conv3 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, groups=self.c1)
        self.conv5 = nn.Conv2d(self.c2, self.c2, kernel_size=5, padding=2, groups=self.c2)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :self.c1, :, :]
        x2 = x[:, self.c1:, :, :]
        out1 = self.conv3(x1)
        out2 = self.conv5(x2)
        out = torch.cat([out1, out2], dim=1)
        return self.act(self.bn(out))


# ==========================================================
# 2. Dual-stream analysis / reconstruction
# ==========================================================
class DualStreamAnalysisHead(nn.Module):
    """
    Fixed dual-stream analysis head.

    Input feature -> temporal Haar decomposition -> low/high split:
      - low-frequency stream  -> projection for BiMamba branch
      - high-frequency stream -> projection for multi-kernel convolution branch
    """
    def __init__(
        self,
        in_channels: int,
        main_channels: int,
        aux_channels: int,
        patch_size: int = 2,
    ):
        super().__init__()
        if main_channels <= 0 or aux_channels <= 0:
            raise ValueError("Both main_channels and aux_channels must be positive.")

        self.wavelet = HaarWaveletTransform1D(in_channels)

        self.main_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                main_channels,
                kernel_size=(1, patch_size),
                stride=(1, patch_size),
            ),
            nn.BatchNorm2d(main_channels),
            nn.ReLU(inplace=True),
        )

        self.aux_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                aux_channels,
                kernel_size=(1, patch_size),
                stride=(1, patch_size),
            ),
            nn.BatchNorm2d(aux_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x_dwt = self.wavelet(x)  # [B, C, 2, T/2, F]
        x_low = x_dwt[:, :, 0, :, :]
        x_high = x_dwt[:, :, 1, :, :]

        main_in = self.main_proj(x_low)
        aux_in = self.aux_proj(x_high)
        return main_in, aux_in


class ReconstructionHead(nn.Module):
    """
    Deconvolution reconstruction head back to residual feature space.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        x = self.proj(x)

        # Residual-size alignment for variable CSI input shapes.
        # Odd spatial sizes after downsampling / reconstruction may lead
        # to a one-step mismatch before residual addition.
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear")
        return x


# ==========================================================
# 3. DualMC core block
# ==========================================================
class DualMCBlock(nn.Module):
    """
    Dual Mamba-Conv block.

    Flow:
      block SE
      -> temporal low/high analysis
      -> low-frequency branch  : BiMambaBlock
      -> high-frequency branch : MultiKernelConv
      -> fusion projection
      -> reconstruction
      -> residual addition
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 2,
        main_channels: int = 32,
        aux_channels: int = 32,
    ):
        super().__init__()
        self.main_channels = main_channels
        self.aux_channels = aux_channels
        self.d_model = main_channels + aux_channels

        self.se = SEBlock(in_channels)

        self.analysis = DualStreamAnalysisHead(
            in_channels=in_channels,
            main_channels=main_channels,
            aux_channels=aux_channels,
            patch_size=patch_size,
        )

        self.main_branch = BiMambaBlock(main_channels, expand=2)
        self.aux_branch = MultiKernelConv(aux_channels)

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
            nn.BatchNorm2d(self.d_model),
        )

        self.reconstruction = ReconstructionHead(
            in_channels=self.d_model,
            out_channels=in_channels,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.se(x)

        main_in, aux_in = self.analysis(x)

        b, c, h, w = main_in.shape
        x_seq = main_in.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_main = self.main_branch(x_seq)
        x_main = x_main.transpose(1, 2).view(b, c, h, w)

        x_aux = self.aux_branch(aux_in)

        x_in = torch.cat([main_in, aux_in], dim=1)
        x_out = torch.cat([x_main, x_aux], dim=1)

        x_fused = x_in + self.fusion_proj(x_out)
        x_recon = self.reconstruction(x_fused, target_size=residual.shape[2:])
        return x_recon + residual


# ==========================================================
# 4. Full model
# ==========================================================
class DualMCN(nn.Module):
    """
    DualMCN: Dual Mamba-Conv Network.

    A lightweight dual-branch architecture for CSI-based fall detection.
    It combines:
      - a BiMamba branch for long-range sequence modeling
      - a multi-kernel convolution branch for local time-frequency pattern extraction

    Final configuration:
      - no stem SE
      - block-level SE enabled
      - temporal Haar low/high analysis enabled
      - dual Mamba-Conv branches enabled
      - deconvolution reconstruction enabled
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 1,
        depths=(1, 1),
        dims=(32, 64),
        patch_size: int = 2,
        stage1_main_channels: int = 32,
        stage1_aux_channels: int = 32,
        stage2_main_channels: int = 64,
        stage2_aux_channels: int = 64,
    ):
        super().__init__()

        if len(depths) != 2:
            raise ValueError(f"depths must have length 2, got {depths}")
        if len(dims) != 2:
            raise ValueError(f"dims must have length 2, got {dims}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.depths = tuple(depths)
        self.dims = tuple(dims)

        # Stem encoder without stem-level SE.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.ModuleList([
            DualMCBlock(
                in_channels=dims[0],
                patch_size=patch_size,
                main_channels=stage1_main_channels,
                aux_channels=stage1_aux_channels,
            )
            for _ in range(depths[0])
        ])

        self.transition = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.ModuleList([
            DualMCBlock(
                in_channels=dims[1],
                patch_size=patch_size,
                main_channels=stage2_main_channels,
                aux_channels=stage2_aux_channels,
            )
            for _ in range(depths[1])
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dims[1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.transition(x)

        for blk in self.stage2:
            x = blk(x)

        x = self.gap(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.classifier(feat)
        return logits


# ==========================================================
# 5. Tiny self-check
# ==========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DualMCN().to(device)
    test_shapes = [
        (2, 1, 500, 232),
        (2, 1, 625, 90),
        (1, 1, 1200, 1026),
    ]

    for shape in test_shapes:
        x = torch.randn(*shape).to(device)
        y = model(x)
        print(f"Input shape: {tuple(x.shape)} -> Output shape: {tuple(y.shape)}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params     : {total_params:.6f} M")
    print(f"Trainable params : {trainable_params:.6f} M")
