import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
import random
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class CSIAugmentation:
    def __init__(self, p=0.7):
        self.p = p

    def gaussian_noise(self, data):
        """
        轻度高斯噪声
        data: [C, T, F]
        """
        noise_std = random.uniform(0.03, 0.08)
        noise = torch.randn_like(data) * noise_std
        return data + noise

    def random_amplitude_scale(self, data):
        """
        轻度全局幅值缩放
        """
        scale = random.uniform(0.85, 1.15)
        return data * scale

    def time_stretching(self, data):
        """
        温和版全局时间拉伸
        data: [C, T, F]
        """
        _, T, Freq = data.shape
        s = random.uniform(0.85, 1.15)
        new_T = max(2, int(T * s))

        data_1d = data.permute(0, 2, 1)  # [C, F, T]
        stretched = F.interpolate(
            data_1d,
            size=new_T,
            mode='linear',
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
        """
        沿时间维做轻度平滑
        避免模型过度依赖过尖锐的源域时间细节
        data: [C, T, F]
        """
        if random.random() < 0.5:
            kernel = torch.tensor([0.25, 0.5, 0.25], device=data.device, dtype=data.dtype)
        else:
            kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=data.device, dtype=data.dtype)

        k = kernel.numel()
        kernel = kernel.view(1, 1, k, 1)

        x = data.unsqueeze(0)   # [1, C, T, F]
        C = x.shape[1]
        kernel = kernel.repeat(C, 1, 1, 1)

        pad = (0, 0, k // 2, k // 2)
        x = F.pad(x, pad, mode="reflect")
        x = F.conv2d(x, kernel, groups=C)

        return x.squeeze(0)

    def time_shift(self, data):
        """
        轻度时间平移
        不改变整体结构，只扰动绝对时间位置
        """
        _, T, _ = data.shape
        max_shift = max(1, int(T * 0.03))   # 最多平移 3%
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(data, shifts=shift, dims=1)

    def band_attenuation(self, data):
        """
        通用连续频带轻度衰减
        不依赖 pair 结构
        data: [C, T, F]
        """
        x = data.clone()
        _, T, Freq = x.shape

        # 连续 band 长度占频率维 8%~20%
        band_len = max(1, int(Freq * random.uniform(0.08, 0.20)))
        start = random.randint(0, max(0, Freq - band_len))
        end = start + band_len

        # 衰减而不是清零
        atten = random.uniform(0.65, 0.9)

        # 有时全程作用，有时只在局部时间段作用
        if random.random() < 0.5:
            x[:, :, start:end] *= atten
        else:
            win_len = max(1, int(T * random.uniform(0.2, 0.5)))
            t0 = random.randint(0, max(0, T - win_len))
            t1 = t0 + win_len
            x[:, t0:t1, start:end] *= atten

        return x

    def local_time_mask(self, data):
        """
        轻度局部时间段扰动
        不是直接置零，而是轻度衰减
        """
        x = data.clone()
        _, T, _ = x.shape

        win_len = max(1, int(T * random.uniform(0.08, 0.20)))
        t0 = random.randint(0, max(0, T - win_len))
        t1 = t0 + win_len

        atten = random.uniform(0.75, 0.95)
        x[:, t0:t1, :] *= atten
        return x

    def __call__(self, sample):
        """
        sample: [C, T, F]
        直接兼容你当前的调用方式
        """
        if random.random() >= self.p:
            return sample

        x = sample.clone()

        # ---- 第一梯队：最贴结论 ----
        if random.random() < 0.5:
            x = self.time_stretching(x)

        if random.random() < 0.5:
            x = self.band_attenuation(x)

        if random.random() < 0.4:
            x = self.time_smoothing(x)

        # ---- 第二梯队：很值得 ----
        if random.random() < 0.4:
            x = self.random_amplitude_scale(x)

        if random.random() < 0.3:
            x = self.gaussian_noise(x)

        # ---- 第三梯队：可选 ----
        if random.random() < 0.25:
            x = self.time_shift(x)

        if random.random() < 0.2:
            x = self.local_time_mask(x)

        return x

class CSIBenchDataset(Dataset):
    def __init__(self, data_root, split_file, task_name="FallDetection", augment=False, target_len=500, target_freq=232):
        self.data_root = data_root
        self.target_len = target_len
        self.target_freq = target_freq
        self.augment = augment
        self.augmentor = CSIAugmentation(p=0.7) if augment else None

        self.task_dir = os.path.join(data_root, task_name)
        if not os.path.exists(self.task_dir):
            candidate = os.path.join(data_root, 'tasks', task_name)
            if os.path.exists(candidate):
                self.task_dir = candidate

        if not os.path.exists(split_file):
            split_file = os.path.join(self.task_dir, 'splits', split_file)

        with open(split_file, 'r') as f:
            valid_ids = set(json.load(f))

        metadata_path = os.path.join(self.task_dir, 'metadata', 'sample_metadata.csv')
        metadata_df = pd.read_csv(metadata_path, dtype={'id': str, 'label': str})
        id_col = 'id' if 'id' in metadata_df.columns else 'sample_id'
        metadata_df = metadata_df[metadata_df[id_col].isin(valid_ids)].reset_index(drop=True)

        label_map_path = os.path.join(self.task_dir, 'metadata', 'label_mapping.json')
        with open(label_map_path, 'r') as f:
            mapping = json.load(f)
            self.label_map = mapping['label_to_idx']
            self.idx_to_label = {int(k): v for k, v in mapping.get('idx_to_label', {}).items()}
            if not self.idx_to_label:
                self.idx_to_label = {v: k for k, v in self.label_map.items()}
            self.num_classes = len(self.label_map)
        self.sorted_keys = sorted(self.label_map.keys(), key=len, reverse=True)

        self.num_samples_original = len(metadata_df)
        self.drop_reason_counts = Counter()
        kept_rows = []
        self.cache = []

        def resolve_label_idx(label_str):
            label_str = str(label_str).strip()
            if label_str in self.label_map:
                return self.label_map[label_str]
            for k in self.sorted_keys:
                if k in label_str:
                    return self.label_map[k]
            return 0

        print(f"[{task_name}] Loaded {self.num_samples_original} samples. Validating & caching...")

        for idx in tqdm(range(len(metadata_df)), desc="Caching"):
            row = metadata_df.iloc[idx]
            rel_path = str(row['file_path']).strip()
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]
            full_path = os.path.join(self.task_dir, rel_path)
            if not os.path.exists(full_path):
                full_path = os.path.join(self.data_root, rel_path)
            if not os.path.exists(full_path):
                self.drop_reason_counts['file_not_found'] += 1
                continue

            try:
                with h5py.File(full_path, 'r') as f:
                    found = False
                    for key in ['CSI_amps', 'csi', 'CSI', 'data']:
                        if key in f:
                            csi_raw = np.array(f[key])
                            found = True
                            break
                    if not found:
                        self.drop_reason_counts['missing_supported_root_key'] += 1
                        continue
            except Exception:
                self.drop_reason_counts['h5_open_error'] += 1
                continue

            try:
                csi_raw = np.nan_to_num(csi_raw, nan=0.0)
                csi_tensor = torch.from_numpy(csi_raw).float()

                if len(csi_tensor.shape) == 2:
                    csi_tensor = csi_tensor.unsqueeze(0)
                elif len(csi_tensor.shape) == 3 and csi_tensor.shape[2] == 1:
                    csi_tensor = csi_tensor.squeeze(2).unsqueeze(0)

                # MTI & Diff
                static = csi_tensor.mean(dim=1, keepdim=True)
                csi_tensor = csi_tensor - static
                diff = csi_tensor[:, 1:, :] - csi_tensor[:, :-1, :]
                zeros = torch.zeros(1, 1, csi_tensor.shape[2], dtype=csi_tensor.dtype)
                csi_tensor = torch.cat([diff, zeros], dim=1)

                # Z-Score
                mean, std = csi_tensor.mean(), csi_tensor.std()
                csi_tensor = (csi_tensor - mean) / (std + 1e-8)

                # Resize/Padding
                final = F.interpolate(
                    csi_tensor.unsqueeze(0),
                    size=(self.target_len, self.target_freq),
                    mode='bilinear'
                ).squeeze(0)
            except Exception:
                self.drop_reason_counts['preprocess_error'] += 1
                continue

            label_idx = resolve_label_idx(row['label'])
            kept_rows.append(row.to_dict())
            self.cache.append((final, label_idx))

        self.df = pd.DataFrame(kept_rows).reset_index(drop=True)
        self.num_samples_kept = len(self.df)
        self.num_samples_dropped = self.num_samples_original - self.num_samples_kept
        kept_counter = Counter(int(label) for _, label in self.cache)
        self.label_distribution_kept = {
            self.idx_to_label.get(idx, str(idx)): int(kept_counter.get(idx, 0))
            for idx in sorted(self.idx_to_label.keys())
        }

        print(
            f"[{task_name}] original={self.num_samples_original} | kept={self.num_samples_kept} | dropped={self.num_samples_dropped}"
        )
        print(f"[{task_name}] drop reasons: {dict(self.drop_reason_counts)}")
        print(f"[{task_name}] kept label distribution: {self.label_distribution_kept}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data, label = self.cache[idx]
        if self.augment and self.augmentor:
            data = self.augmentor(data.clone())
        return data, label

def robust_collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch: return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)