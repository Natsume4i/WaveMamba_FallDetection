import os
import json
import torch
import pandas as pd
import numpy as np
import re
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import torch

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

        _, T, _ = data.shape
        max_shift = max(1, int(T * 0.03))   # 最多平移 3%
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(data, shifts=shift, dims=1)

    def band_attenuation(self, data):

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
        """
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


class CSIDataset(Dataset):
    def __init__(
        self,
        data_root,
        split_file,
        mapping_file,
        task_type='binary',
        target_len=1200,
        target_subcarrier=114,
        augment=False
    ):
        self.data_root = data_root
        self.target_len = target_len
        self.target_subcarrier = target_subcarrier
        self.task_type = task_type
        self.augment = augment
        self.augmentor = CSIAugmentation(p=0.7) if augment else None

        self.num_pairs = 9
        self.num_subcarriers = 114
        self.original_feature_dim = self.num_pairs * self.num_subcarriers
        self.target_freq_dim = self.num_pairs * self.target_subcarrier

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split {split_file} not found")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping {mapping_file} not found")

        with open(split_file, 'r') as f:
            self.file_list = json.load(f)
        with open(mapping_file, 'r') as f:
            self.mapping_config = json.load(f)

        if task_type == 'binary':
            self.label_map = self.mapping_config['binary_mapping']['label_to_idx']
        else:
            self.label_map = self.mapping_config['multiclass_mapping']['label_to_idx']
        self.num_classes = len(self.label_map)

        print(
            f"[{task_type.upper()}] Preloading {len(self.file_list)} samples to RAM... "
            f"(target_len={self.target_len}, target_subcarrier={self.target_subcarrier}, "
            f"target_freq_dim={self.target_freq_dim})"
        )

        self.cache = []
        for filename in tqdm(self.file_list, desc="Caching"):
            data_tensor, label, raw_action_name = self.load_and_process_file(filename)
            if data_tensor is not None:
                self.cache.append((data_tensor, label, raw_action_name))

        print(f"Done! Cached {len(self.cache)} valid samples.")

    def get_label_from_filename(self, filename):
        """解析文件名获取标签"""
        try:
            binary_char = filename.split('_')[-1].split('.')[0]
            binary_idx = int(binary_char)
        except Exception:
            binary_idx = 0

        match = re.search(r'v\d+_(.*)_\d\.csv', filename, re.IGNORECASE)
        raw_action_name = match.group(1).lower() if match else "unknown"

        multi_map = self.mapping_config["multiclass_mapping"]["label_to_idx"]

        if "fall" in raw_action_name or "syncope" in raw_action_name:
            final_name = "fall"
        elif "sit" in raw_action_name:
            final_name = "sit"
        else:
            final_name = raw_action_name

        multi_idx = multi_map.get(final_name, -1)
        return binary_idx, multi_idx, raw_action_name

    def reduce_frequency_per_pair(self, csi_tensor):
        """
        输入:
            csi_tensor: [1, T, 1026]

        原始频率维顺序:
            [subcarrier_0 的 9 个天线对, subcarrier_1 的 9 个天线对, ...]

        重排后:
            [1, T, 9, 114]

        若 target_subcarrier == 114:
            只重排，不插值

        若 target_subcarrier != 114:
            对每个天线对内部的 114 子载波做线性插值
        """
        b, t, fdim = csi_tensor.shape
        if fdim != self.original_feature_dim:
            raise ValueError(
                f"Expected feature dim {self.original_feature_dim}, got {fdim}"
            )

        # [1, T, 1026] -> [1, T, 114, 9]
        x = csi_tensor.view(b, t, self.num_subcarriers, self.num_pairs)

        # [1, T, 114, 9] -> [1, T, 9, 114]
        x = x.permute(0, 1, 3, 2).contiguous()

        # 如果目标仍然是 114，则只做重排，不做插值
        if self.target_subcarrier == self.num_subcarriers:
            x = x.view(b, t, self.num_pairs * self.num_subcarriers)
            return x

        # [1, T, 9, 114] -> [b*t*9, 1, 114]
        x = x.view(b * t * self.num_pairs, 1, self.num_subcarriers)

        # 线性插值: 114 -> target_subcarrier
        x = F.interpolate(
            x,
            size=self.target_subcarrier,
            mode='linear',
            align_corners=False
        )

        # [b*t*9, 1, target_subcarrier] -> [1, T, 9, target_subcarrier]
        x = x.view(b, t, self.num_pairs, self.target_subcarrier)

        # [1, T, 9, target_subcarrier] -> [1, T, 9 * target_subcarrier]
        x = x.view(b, t, self.num_pairs * self.target_subcarrier)

        return x

    def fix_time_length(self, csi_tensor):
        """
        输入:
            csi_tensor: [1, T, F]

        输出:
            [1, target_len, F]

        策略:
            - T == target_len: 直接返回
            - T > target_len : center crop
            - T < target_len : 尾部补零
        """
        _, t, f = csi_tensor.shape

        if t == self.target_len:
            return csi_tensor

        if t > self.target_len:
            start = (t - self.target_len) // 2
            end = start + self.target_len
            return csi_tensor[:, start:end, :]

        # t < target_len
        pad_len = self.target_len - t
        pad_tensor = torch.zeros(
            (1, pad_len, f),
            dtype=csi_tensor.dtype,
            device=csi_tensor.device
        )
        return torch.cat([csi_tensor, pad_tensor], dim=1)

    def load_and_process_file(self, filename):
        file_path = os.path.join(self.data_root, filename)

        try:
            df = pd.read_csv(file_path)
            csi_data = df.filter(like='subcarrier').values.astype(np.float32)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None, None, None

        csi_tensor = torch.from_numpy(csi_data)  # [T, 1026]

        if csi_tensor.ndim != 2:
            print(f"Unexpected shape in {filename}: {tuple(csi_tensor.shape)}")
            return None, None, None

        # [T, 1026] -> [1, T, 1026]
        csi_tensor = csi_tensor.unsqueeze(0)

        # 1) 频率维重排；target_subcarrier != 114 时才做频率插值
        csi_tensor = self.reduce_frequency_per_pair(csi_tensor)  # [1, T, 9*target_subcarrier]

        # 2) 去静态分量
        static = csi_tensor.mean(dim=1, keepdim=True)
        csi_tensor = csi_tensor - static

        # 3) 时间差分 + 末尾补零 + abs
        diff = csi_tensor[:, 1:, :] - csi_tensor[:, :-1, :]
        zeros = torch.zeros(
            1, 1, csi_tensor.shape[2],
            dtype=csi_tensor.dtype,
            device=csi_tensor.device
        )
        csi_tensor = torch.cat([diff, zeros], dim=1)
        # 4) 不做样本级 z-score

        # 5) 时间维统一：padding / center crop，不做时间插值
        csi_tensor = self.fix_time_length(csi_tensor)  # [1, target_len, target_freq_dim]

        # 6) 标签
        bin_label, multi_label, raw_action_name = self.get_label_from_filename(filename)
        label = bin_label if self.task_type == 'binary' else multi_label

        if label == -1:
            print(f"Warning: invalid label for {filename}, skip.")
            return None, None, None

        return csi_tensor, torch.tensor(label, dtype=torch.long), raw_action_name

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        data_tensor, label, raw_name = self.cache[idx]
        if self.augment and self.augmentor:
            data_tensor = self.augmentor(data_tensor.clone())
        return data_tensor, label, raw_name

def robust_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return (
            torch.empty(0, 1, 0, 0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            []
        )
    return torch.utils.data.dataloader.default_collate(batch)