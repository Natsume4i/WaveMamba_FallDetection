
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
# Small helpers
# ==========================================================
def make_norm2d(num_channels: int, use_bn: bool) -> nn.Module:
    return nn.BatchNorm2d(num_channels) if use_bn else nn.Identity()


def make_se(channels: int, use_se: bool) -> nn.Module:
    return SEBlock(channels) if use_se else nn.Identity()


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


class VimBlock(nn.Module):
    """
    Bidirectional Mamba wrapper with a single internal LayerNorm.
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
    Split channels into two parts:
      - first half: 3x3 depthwise conv
      - second half: 5x5 depthwise conv
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


class MultiKernelConvPW(nn.Module):
    """
    Multi-kernel depthwise conv + pointwise fusion.
    Used for aux branch ablation with explicit channel interaction.
    """
    def __init__(self, channels: int):
        super().__init__()
        if channels < 2:
            raise ValueError(f"MultiKernelConvPW requires channels >= 2, got {channels}")

        self.c1 = channels // 2
        self.c2 = channels - self.c1

        self.conv3 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, groups=self.c1)
        self.conv5 = nn.Conv2d(self.c2, self.c2, kernel_size=5, padding=2, groups=self.c2)

        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.SiLU()
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :self.c1, :, :]
        x2 = x[:, self.c1:, :, :]
        out1 = self.conv3(x1)
        out2 = self.conv5(x2)
        out = torch.cat([out1, out2], dim=1)
        out = self.act1(self.bn1(out))
        out = self.act2(self.bn2(self.pw(out)))
        return out


# ==========================================================
# 2. Dual-stream analysis / reconstruction
# ==========================================================
class DualStreamAnalysisHead(nn.Module):
    """
    Explicit dual-stream analysis head.

    use_wavelet=True:
      x -> Haar -> low/high split
      low  -> main_proj
      high -> aux_proj

    use_wavelet=False:
      x -> main_direct_proj
      x -> aux_direct_proj
    """
    def __init__(
        self,
        in_channels: int,
        main_channels: int,
        aux_channels: int,
        patch_size: int = 2,
        use_wavelet: bool = True,
        use_main_proj_bn: bool = True,
        use_aux_proj_bn: bool = True,
    ):
        super().__init__()
        if main_channels <= 0 and aux_channels <= 0:
            raise ValueError("At least one of main_channels or aux_channels must be > 0")

        self.use_wavelet = use_wavelet

        if use_wavelet:
            self.wavelet = HaarWaveletTransform1D(in_channels)
            main_kernel = (1, patch_size)
            main_stride = (1, patch_size)
            aux_kernel = (1, patch_size)
            aux_stride = (1, patch_size)
        else:
            self.wavelet = None
            # Match wavelet case output size: T -> T/2, F -> F/patch_size
            main_kernel = (2, patch_size)
            main_stride = (2, patch_size)
            aux_kernel = (2, patch_size)
            aux_stride = (2, patch_size)

        if main_channels > 0:
            self.main_proj = nn.Sequential(
                nn.Conv2d(in_channels, main_channels, kernel_size=main_kernel, stride=main_stride),
                make_norm2d(main_channels, use_main_proj_bn),
                nn.ReLU(inplace=True),
            )
        else:
            self.main_proj = None

        if aux_channels > 0:
            self.aux_proj = nn.Sequential(
                nn.Conv2d(in_channels, aux_channels, kernel_size=aux_kernel, stride=aux_stride),
                make_norm2d(aux_channels, use_aux_proj_bn),
                nn.ReLU(inplace=True),
            )
        else:
            self.aux_proj = None

    def forward(self, x: torch.Tensor):
        if self.use_wavelet:
            x_dwt = self.wavelet(x)  # [B, C, 2, T/2, F]
            x_low = x_dwt[:, :, 0, :, :]
            x_high = x_dwt[:, :, 1, :, :]
        else:
            x_low = x
            x_high = x

        main_in = self.main_proj(x_low) if self.main_proj is not None else None
        aux_in = self.aux_proj(x_high) if self.aux_proj is not None else None
        return main_in, aux_in


class ReconstructionHead(nn.Module):
    """
    Unified reconstruction head back to residual space.

    recon_type:
      - "deconv"
      - "interp_1x1"
      - "interp_3x3"
      - "none"      -> minimal projection path
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        recon_type: str = "deconv",
        use_upsample_bn: bool = True,
    ):
        super().__init__()
        valid = {"deconv", "interp_1x1", "interp_3x3", "none"}
        if recon_type not in valid:
            raise ValueError(f"recon_type must be one of {valid}, got {recon_type}")

        self.recon_type = recon_type

        if recon_type == "deconv":
            self.proj = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
                make_norm2d(out_channels, use_upsample_bn),
                nn.ReLU(inplace=True),
            )
        elif recon_type == "interp_1x1":
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm2d(out_channels, use_upsample_bn),
                nn.ReLU(inplace=True),
            )
        elif recon_type == "interp_3x3":
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                make_norm2d(out_channels, use_upsample_bn),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        if self.recon_type == "deconv":
            x = self.proj(x)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear")
            return x

        x = F.interpolate(x, size=target_size, mode="bilinear")
        x = self.proj(x)
        return x


# ==========================================================
# 3. WaveMamba core block (explicit dual-stream)
# ==========================================================
class WaveMambaBlock(nn.Module):
    """
    Core block:
      (optional block SE)
      -> explicit low/high split
      -> low  -> main branch
      -> high -> aux branch
      -> fusion
      -> ReconstructionHead
      -> residual add
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 2,
        main_channels: int = 32,
        aux_channels: int = 32,
        use_block_se: bool = False,
        use_main_proj_bn: bool = True,
        use_aux_proj_bn: bool = True,
        use_fusion_proj_bn: bool = True,
        use_upsample_bn: bool = True,
        use_wavelet: bool = True,
        branch_mode: str = "dual",
        recon_type: str = "deconv",
        aux_variant: str = "mkconv",
    ):
        super().__init__()

        valid_branch_modes = {"dual", "main_only", "aux_only"}
        if branch_mode not in valid_branch_modes:
            raise ValueError(f"branch_mode must be one of {valid_branch_modes}, got {branch_mode}")

        valid_aux_variants = {"mkconv", "mkconv_pw"}
        if aux_variant not in valid_aux_variants:
            raise ValueError(f"aux_variant must be one of {valid_aux_variants}, got {aux_variant}")

        self.branch_mode = branch_mode
        self.main_channels = main_channels if branch_mode != "aux_only" else 0
        self.aux_channels = aux_channels if branch_mode != "main_only" else 0
        self.d_model = self.main_channels + self.aux_channels

        if self.d_model <= 0:
            raise ValueError("WaveMambaBlock requires at least one active branch")

        self.se = make_se(in_channels, use_block_se)

        self.analysis = DualStreamAnalysisHead(
            in_channels=in_channels,
            main_channels=self.main_channels,
            aux_channels=self.aux_channels,
            patch_size=patch_size,
            use_wavelet=use_wavelet,
            use_main_proj_bn=use_main_proj_bn,
            use_aux_proj_bn=use_aux_proj_bn,
        )

        self.main_branch = VimBlock(self.main_channels, expand=2) if self.main_channels > 0 else None

        if self.aux_channels > 0:
            if aux_variant == "mkconv":
                self.aux_branch = MultiKernelConv(self.aux_channels)
            else:
                self.aux_branch = MultiKernelConvPW(self.aux_channels)
        else:
            self.aux_branch = None

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
            make_norm2d(self.d_model, use_fusion_proj_bn),
        )

        self.reconstruction = ReconstructionHead(
            in_channels=self.d_model,
            out_channels=in_channels,
            patch_size=patch_size,
            recon_type=recon_type,
            use_upsample_bn=use_upsample_bn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.se(x)

        main_in, aux_in = self.analysis(x)

        fused_inputs = []
        fused_outputs = []

        if self.main_branch is not None:
            b, c, h, w = main_in.shape
            x_seq = main_in.flatten(2).transpose(1, 2)
            x_main = self.main_branch(x_seq)
            x_main = x_main.transpose(1, 2).view(b, c, h, w)
            fused_inputs.append(main_in)
            fused_outputs.append(x_main)

        if self.aux_branch is not None:
            x_aux = self.aux_branch(aux_in)
            fused_inputs.append(aux_in)
            fused_outputs.append(x_aux)

        x_in = torch.cat(fused_inputs, dim=1)
        x_out = torch.cat(fused_outputs, dim=1)

        x_fused = x_in + self.fusion_proj(x_out)
        x_recon = self.reconstruction(x_fused, target_size=residual.shape[2:])
        return x_recon + residual


# ==========================================================
# 4. Full model
# ==========================================================
class WaveMamba(nn.Module):
    """
    Final formal implementation based on explicit dual-stream frequency routing.

    Default configuration:
      - depths=(1, 1)
      - dims=(32, 64)
      - patch_size=2
      - stage1_main_channels=32
      - stage1_aux_channels=32
      - stage2_main_channels=64
      - stage2_aux_channels=64
      - use_stem_se=True
      - use_block_se=False
      - all key BN enabled
      - use_wavelet=True
      - branch_mode="dual"
      - recon_type="deconv"
      - aux_variant="mkconv"
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
        use_stem_se: bool = True,
        use_block_se: bool = False,
        use_main_proj_bn: bool = True,
        use_aux_proj_bn: bool = True,
        use_fusion_proj_bn: bool = True,
        use_upsample_bn: bool = True,
        use_wavelet: bool = True,
        branch_mode: str = "dual",
        recon_type: str = "deconv",
        aux_variant: str = "mkconv",
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

        stem_se = make_se(dims[0], use_stem_se)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            stem_se,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), padding=1),
        )

        self.stage1 = nn.ModuleList([
            WaveMambaBlock(
                in_channels=dims[0],
                patch_size=patch_size,
                main_channels=stage1_main_channels,
                aux_channels=stage1_aux_channels,
                use_block_se=use_block_se,
                use_main_proj_bn=use_main_proj_bn,
                use_aux_proj_bn=use_aux_proj_bn,
                use_fusion_proj_bn=use_fusion_proj_bn,
                use_upsample_bn=use_upsample_bn,
                use_wavelet=use_wavelet,
                branch_mode=branch_mode,
                recon_type=recon_type,
                aux_variant=aux_variant,
            )
            for _ in range(depths[0])
        ])

        self.transition = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.ModuleList([
            WaveMambaBlock(
                in_channels=dims[1],
                patch_size=patch_size,
                main_channels=stage2_main_channels,
                aux_channels=stage2_aux_channels,
                use_block_se=use_block_se,
                use_main_proj_bn=use_main_proj_bn,
                use_aux_proj_bn=use_aux_proj_bn,
                use_fusion_proj_bn=use_fusion_proj_bn,
                use_upsample_bn=use_upsample_bn,
                use_wavelet=use_wavelet,
                branch_mode=branch_mode,
                recon_type=recon_type,
                aux_variant=aux_variant,
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

    model = WaveMamba().to(device)
    x = torch.randn(2, 1, 500, 232).to(device)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params     : {total_params:.6f} M")
    print(f"Trainable params : {trainable_params:.6f} M")
