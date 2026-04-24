import os
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================
# 输入: [C, T, F]
# ============================================================
class CSIAugmentation:
    def __init__(self, p=0.7):
        self.p = p

    def gaussian_noise(self, data):
        noise_std = random.uniform(0.03, 0.08)
        noise = torch.randn_like(data) * noise_std
        return data + noise

    def random_amplitude_scale(self, data):
        scale = random.uniform(0.85, 1.15)
        return data * scale

    def time_stretching(self, data):
        _, T, Freq = data.shape
        s = random.uniform(0.85, 1.15)
        new_T = max(2, int(T * s))

        data_1d = data.permute(0, 2, 1)  # [C, F, T]
        stretched = F.interpolate(
            data_1d,
            size=new_T,
            mode="linear",
            align_corners=False
        )
        stretched = stretched.permute(0, 2, 1)  # [C, new_T, F]

        if new_T > T:
            start = (new_T - T) // 2
            return stretched[:, start:start + T, :]
        else:
            pad_left = (T - new_T) // 2
            pad_right = T - new_T - pad_left
            return F.pad(stretched, (0, 0, pad_left, pad_right), mode="constant", value=0)

    def time_smoothing(self, data):
        if random.random() < 0.5:
            kernel = torch.tensor([0.25, 0.5, 0.25], device=data.device, dtype=data.dtype)
        else:
            kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=data.device, dtype=data.dtype)

        k = kernel.numel()
        kernel = kernel.view(1, 1, k, 1)

        x = data.unsqueeze(0)  # [1, C, T, F]
        C = x.shape[1]
        kernel = kernel.repeat(C, 1, 1, 1)

        pad = (0, 0, k // 2, k // 2)
        x = F.pad(x, pad, mode="reflect")
        x = F.conv2d(x, kernel, groups=C)

        return x.squeeze(0)

    def time_shift(self, data):
        _, T, _ = data.shape
        max_shift = max(1, int(T * 0.03))
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(data, shifts=shift, dims=1)

    def band_attenuation(self, data):
        x = data.clone()
        _, T, Freq = x.shape

        band_len = max(1, int(Freq * random.uniform(0.08, 0.20)))
        start = random.randint(0, max(0, Freq - band_len))
        end = start + band_len

        atten = random.uniform(0.65, 0.9)

        if random.random() < 0.5:
            x[:, :, start:end] *= atten
        else:
            win_len = max(1, int(T * random.uniform(0.2, 0.5)))
            t0 = random.randint(0, max(0, T - win_len))
            t1 = t0 + win_len
            x[:, t0:t1, start:end] *= atten

        return x

    def local_time_mask(self, data):
        x = data.clone()
        _, T, _ = x.shape

        win_len = max(1, int(T * random.uniform(0.08, 0.20)))
        t0 = random.randint(0, max(0, T - win_len))
        t1 = t0 + win_len

        atten = random.uniform(0.75, 0.95)
        x[:, t0:t1, :] *= atten
        return x

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        x = sample.clone()

        if random.random() < 0.5:
            x = self.time_stretching(x)

        if random.random() < 0.5:
            x = self.band_attenuation(x)

        if random.random() < 0.4:
            x = self.time_smoothing(x)

        if random.random() < 0.4:
            x = self.random_amplitude_scale(x)

        if random.random() < 0.3:
            x = self.gaussian_noise(x)

        if random.random() < 0.25:
            x = self.time_shift(x)

        if random.random() < 0.2:
            x = self.local_time_mask(x)

        return x


# ============================================================
# ENetFall Dataset
# ============================================================
class ENetFallDataset(Dataset):
    """
    ENetFall 数据集加载器

    假定 mat 文件中至少包含:
      - dataset_CSI_t
      - dataset_labels

    支持两种常见 shape:
      - (N, 625, 90)
      - (N, 90, 625)

    最终输出:
      x:     [1, 625, 90]
      label: long, 0=Nonfall, 1=Fall
    """

    DATA_KEY = "dataset_CSI_t"
    LABEL_KEY = "dataset_labels"

    def __init__(self, data_root, file_list, augment=False):
        self.data_root = data_root
        self.file_list = file_list
        self.augment = augment
        self.augmentor = CSIAugmentation(p=0.7) if augment else None
        self.cache = []   # list of (tensor[1,625,90], label:int)

        print(f"[ENetFall] Loading files: {file_list}")

        for fname in file_list:
            path = os.path.join(data_root, fname)
            if not os.path.exists(path):
                print(f"  [Warn] File not found: {path}")
                continue

            try:
                mat = sio.loadmat(path)
            except Exception as e:
                print(f"  [Warn] Failed to load {fname}: {e}")
                continue

            # ---------- key 检查 ----------
            if self.DATA_KEY not in mat:
                print(f"  [Warn] Missing key '{self.DATA_KEY}' in {fname}, skip")
                continue
            if self.LABEL_KEY not in mat:
                print(f"  [Warn] Missing key '{self.LABEL_KEY}' in {fname}, skip")
                continue

            data = mat[self.DATA_KEY]
            labels = mat[self.LABEL_KEY]

            # ---------- 基本 shape 检查 ----------
            if not isinstance(data, np.ndarray) or data.ndim != 3:
                print(f"  [Warn] Bad data shape in {fname}: {getattr(data, 'shape', None)}, skip")
                continue

            if not isinstance(labels, np.ndarray):
                print(f"  [Warn] Bad labels type in {fname}, skip")
                continue

            # ---------- 自动转成 (N, 625, 90) ----------
            if data.shape[1:] == (625, 90):
                pass
            elif data.shape[1:] == (90, 625):
                data = np.transpose(data, (0, 2, 1))
            else:
                print(f"  [Warn] Unsupported data shape in {fname}: {data.shape}, skip")
                continue

            labels = labels.squeeze()
            if labels.ndim == 0:
                labels = np.array([labels])

            # ---------- label 长度检查 ----------
            if len(labels) != len(data):
                print(f"  [Warn] Label length mismatch in {fname}: data={len(data)} labels={len(labels)}, skip")
                continue

            # ---------- nan / inf 清理 ----------
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # ---------- 标签统计 ----------
            n_fall = int((labels == 1).sum())
            n_nonfall = int((labels == 0).sum())
            print(f"  {fname}: {len(labels)} samples | Fall={n_fall} Nonfall={n_nonfall}")

            # ---------- 逐样本处理 ----------
            valid_count = 0
            for i in range(len(labels)):
                try:
                    x = torch.from_numpy(data[i]).float()   # [625, 90]
                    label = int(labels[i])

                    if label not in [0, 1]:
                        print(f"    [Warn] Invalid label {label} in {fname}[{i}], skip")
                        continue

                    # 1. 去静态
                    static = x.mean(dim=0, keepdim=True)
                    x = x - static

                    # 2. 差分
                    diff = x[1:, :] - x[:-1, :]

                    # 3. 补最后一帧并取绝对值
                    zeros = torch.zeros(1, x.shape[1], dtype=x.dtype)
                    x = torch.cat([diff, zeros], dim=0)

                    # 4. 全样本 z-score
                    mean = x.mean()
                    std = x.std()
                    if std < 1e-8:
                        std = torch.tensor(1.0, dtype=x.dtype)
                    x = (x - mean) / (std + 1e-8)

                    # 4. 按子载波独立归一化
                    #mean = x.mean(dim=0, keepdim=True)   # [1, F]
                    #std = x.std(dim=0, keepdim=True)     # [1, F]
                    #std = torch.where(std < 1e-8, torch.ones_like(std), std)
                    #x = (x - mean) / (std + 1e-8)

                    # 5. [1, 625, 90]
                    x = x.unsqueeze(0)

                    # 最终安全检查
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        print(f"    [Warn] NaN/Inf after preprocess in {fname}[{i}], skip")
                        continue

                    self.cache.append((x, label))
                    valid_count += 1

                except Exception as e:
                    print(f"    [Warn] Failed sample {fname}[{i}]: {e}")
                    continue

            print(f"    valid cached from {fname}: {valid_count}")

        print(f"[ENetFall] Total cached samples: {len(self.cache)}")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        x, label = self.cache[idx]
        x = x.clone()
        if self.augment and self.augmentor is not None:
            x = self.augmentor(x)
        return x, torch.tensor(label, dtype=torch.long)


# ============================================================
# 更稳的 collate_fn
# ============================================================
def robust_collate_fn(batch):
    """
    过滤掉 None 或坏样本，避免 dataloader 崩掉
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return (
            torch.empty(0, 1, 625, 90),
            torch.empty(0, dtype=torch.long)
        )
    return torch.utils.data.dataloader.default_collate(batch)