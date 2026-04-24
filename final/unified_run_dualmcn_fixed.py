import os
import json
import copy
import random
import argparse
import warnings
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dualmcn import DualMCN
from dataset_loader_csibench import CSIBenchDataset, robust_collate_fn as csibench_collate_fn
from enetfall_dataset import ENetFallDataset, robust_collate_fn as enetfall_collate_fn
from dataset_loader_ourdata import CSIDataset, robust_collate_fn as ourdata_collate_fn


# =========================================================
# Randomness
# =========================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# EMA
# =========================================================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for k, v in ema_state.items():
            model_v = model_state[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)
        return self


# =========================================================
# Small helpers
# =========================================================
def count_parameters(model: nn.Module) -> Tuple[float, float]:
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return float(total), float(trainable)


def format_mean_std(m: Optional[float], s: Optional[float], digits: int = 4) -> str:
    if m is None:
        return "-"
    return f"{float(m):.{digits}f} ± {float(0.0 if s is None else s):.{digits}f}"


def aggregate_numeric_dicts(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    out: Dict[str, float] = {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    for k in keys:
        vals = [float(x[k]) for x in rows if k in x and isinstance(x[k], (int, float))]
        if vals:
            out[f"{k}_mean"] = float(mean(vals))
            out[f"{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_load(path: str, device: torch.device):
    return torch.load(path, map_location=device)


# =========================================================
# Config validation
# =========================================================
VALID_MAIN_MODELS = ["dualmcn_default", "resnet18", "lstm", "gru", "tcn", "vim"]
VALID_ABLATION_MODELS = [
    "dualmcn_default",

    # Ablations based on the new default:
    # no stem SE + block SE
    "dualmcn_no_wavelet",
    "dualmcn_main_only",
    "dualmcn_aux_only",
    "dualmcn_recon_interp_1x1",
    "dualmcn_aux_pw",

    # The old default and its ablations:
    # stem SE + no block SE
    "dualmcn_stem_se_no_block_se",
    "dualmcn_no_wavelet_stem_se_no_block_se",
    "dualmcn_main_only_stem_se_no_block_se",
    "dualmcn_aux_only_stem_se_no_block_se",
    "dualmcn_recon_interp_1x1_stem_se_no_block_se",
    "dualmcn_aux_pw_stem_se_no_block_se",

    # Independent SE ablation
    "dualmcn_no_stem_se_no_block_se",
]
VALID_CHECKPOINT_POLICIES = ["valbest_ema", "valbest_raw", "top5avg_bnrecal"]
VALID_DATASETS = ["csibench", "enetfall", "ourdata"]


def load_baseline_registry():
    # Lazy import: some environments may have torchvision/Vim dependency issues.
    from baseline_models import (
        ResNet18Baseline,
        BiLSTMBaseline,
        GRUBaseline,
        TCNBaseline,
        VimBaseline,
    )
    return {
        "resnet18": ResNet18Baseline,
        "lstm": BiLSTMBaseline,
        "gru": GRUBaseline,
        "tcn": TCNBaseline,
        "vim": VimBaseline,
    }


def validate_config(cfg: Dict[str, Any]):
    dataset_name = cfg["dataset"]["name"]
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Unsupported dataset.name: {dataset_name}")

    policy = cfg["checkpoint"]["checkpoint_policy"]
    if policy not in VALID_CHECKPOINT_POLICIES:
        raise ValueError(f"Unsupported checkpoint_policy: {policy}")

    for model_name in cfg["models"]["models_main"]:
        if model_name not in VALID_MAIN_MODELS:
            raise ValueError(f"Unsupported models_main entry: {model_name}")

    for model_name in cfg["models"]["models_ablation"]:
        if model_name not in VALID_ABLATION_MODELS:
            raise ValueError(f"Unsupported models_ablation entry: {model_name}")

    for exp_name, exp_cfg in cfg["experiments"].items():
        if not isinstance(exp_cfg.get("enabled", True), bool):
            raise ValueError(f"experiments.{exp_name}.enabled must be bool")
        for model_name in exp_cfg.get("models", []):
            if model_name not in VALID_MAIN_MODELS + VALID_ABLATION_MODELS:
                raise ValueError(f"Unsupported model '{model_name}' in experiments.{exp_name}")


# =========================================================
# ENetFall helper datasets
# =========================================================
class ENetFallCacheSubset(Dataset):
    def __init__(self, base_dataset: ENetFallDataset, indices: List[int], augment: bool = False):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.augment = augment
        self.augmentor = base_dataset.augmentor if augment else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, label = self.base_dataset.cache[real_idx]
        x = x.clone()
        if self.augment and self.augmentor is not None:
            x = self.augmentor(x)
        return x, torch.tensor(label, dtype=torch.long)


class ConcatSimpleDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cum_sizes: List[int] = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx):
        prev = 0
        for ds, end in zip(self.datasets, self.cum_sizes):
            if idx < end:
                return ds[idx - prev]
            prev = end
        raise IndexError(idx)


# =========================================================
# Dataset builders
# =========================================================
def build_datasets(cfg: Dict[str, Any], use_augmentation: bool, seed: int):
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]

    if dataset_name == "csibench":
        train_ds = CSIBenchDataset(
            data_root=dataset_cfg["data_root"],
            task_name=dataset_cfg.get("task_name", "FallDetection"),
            split_file=dataset_cfg["train_split"],
            augment=use_augmentation,
            target_len=dataset_cfg["target_len"],
            target_freq=dataset_cfg["target_freq"],
        )
        val_ds = CSIBenchDataset(
            data_root=dataset_cfg["data_root"],
            task_name=dataset_cfg.get("task_name", "FallDetection"),
            split_file=dataset_cfg["val_split"],
            augment=False,
            target_len=dataset_cfg["target_len"],
            target_freq=dataset_cfg["target_freq"],
        )
        test_id_ds = None
        test_cross_env_ds = None
        eval_mode = dataset_cfg.get("eval_mode", "cross_env")
        if eval_mode in ["id", "both"]:
            test_id_ds = CSIBenchDataset(
                data_root=dataset_cfg["data_root"],
                task_name=dataset_cfg.get("task_name", "FallDetection"),
                split_file=dataset_cfg["test_id_split"],
                augment=False,
                target_len=dataset_cfg["target_len"],
                target_freq=dataset_cfg["target_freq"],
            )
        if eval_mode in ["cross_env", "both"]:
            test_cross_env_ds = CSIBenchDataset(
                data_root=dataset_cfg["data_root"],
                task_name=dataset_cfg.get("task_name", "FallDetection"),
                split_file=dataset_cfg["test_cross_env_split"],
                augment=False,
                target_len=dataset_cfg["target_len"],
                target_freq=dataset_cfg["target_freq"],
            )
        idx_to_label = {int(v): str(k) for k, v in train_ds.label_map.items()}
        return {
            "train": train_ds,
            "val": val_ds,
            "test_id": test_id_ds,
            "test_cross_env": test_cross_env_ds,
            "idx_to_label": idx_to_label,
            "collate_fn": csibench_collate_fn,
            "input_h": dataset_cfg["target_len"],
            "input_w": dataset_cfg["target_freq"],
        }

    if dataset_name == "enetfall":
        train_subsets = []
        val_subsets = []
        env_stats = {}
        for env_name in dataset_cfg["train_envs"]:
            base_env_ds = ENetFallDataset(
                data_root=dataset_cfg["data_root"],
                file_list=[env_name],
                augment=use_augmentation,
            )
            labels = [label for _, label in base_env_ds.cache]
            indices = np.arange(len(labels))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=dataset_cfg["val_ratio"],
                random_state=seed,
                stratify=labels,
                shuffle=True,
            )
            train_subsets.append(ENetFallCacheSubset(base_env_ds, train_idx.tolist(), augment=use_augmentation))
            val_subsets.append(ENetFallCacheSubset(base_env_ds, val_idx.tolist(), augment=False))
            env_stats[env_name] = {
                "total": len(labels),
                "fall": int(np.sum(np.array(labels) == 1)),
                "nonfall": int(np.sum(np.array(labels) == 0)),
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
            }

        train_ds = ConcatSimpleDataset(train_subsets)
        val_ds = ConcatSimpleDataset(val_subsets)
        test_living_room_ds = ENetFallDataset(
            data_root=dataset_cfg["data_root"],
            file_list=[dataset_cfg["test_envs"]["living_room"]],
            augment=False,
        )
        test_lecture_room_ds = ENetFallDataset(
            data_root=dataset_cfg["data_root"],
            file_list=[dataset_cfg["test_envs"]["lecture_room"]],
            augment=False,
        )
        return {
            "train": train_ds,
            "val": val_ds,
            "test_living_room": test_living_room_ds,
            "test_lecture_room": test_lecture_room_ds,
            "idx_to_label": {0: "Nonfall", 1: "Fall"},
            "collate_fn": enetfall_collate_fn,
            "input_h": dataset_cfg["input_h"],
            "input_w": dataset_cfg["input_w"],
            "env_stats": env_stats,
        }

    if dataset_name == "ourdata":
        mapping_file = os.path.join(dataset_cfg["meta_root"], dataset_cfg["mapping_file"])
        train_split = os.path.join(dataset_cfg["meta_root"], dataset_cfg["train_split"])
        val_split = os.path.join(dataset_cfg["meta_root"], dataset_cfg["val_split"])
        test_cross_env_split = os.path.join(dataset_cfg["meta_root"], dataset_cfg["test_cross_env_split"])

        common_kwargs = dict(
            data_root=dataset_cfg["data_root"],
            mapping_file=mapping_file,
            task_type="binary",
            target_len=dataset_cfg["target_len"],
            target_subcarrier=dataset_cfg["target_subcarrier"],
        )
        train_ds = CSIDataset(split_file=train_split, augment=use_augmentation, **common_kwargs)
        val_ds = CSIDataset(split_file=val_split, augment=False, **common_kwargs)
        test_cross_env_ds = CSIDataset(split_file=test_cross_env_split, augment=False, **common_kwargs)
        return {
            "train": train_ds,
            "val": val_ds,
            "test_cross_env": test_cross_env_ds,
            "idx_to_label": {0: "Nonfall", 1: "Fall"},
            "collate_fn": ourdata_collate_fn,
            "input_h": dataset_cfg["input_h"],
            "input_w": dataset_cfg["input_w"],
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# =========================================================
# Loaders
# =========================================================
def build_loaders(cfg: Dict[str, Any], datasets: Dict[str, Any], seed: int):
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 0)
    collate_fn = datasets["collate_fn"]

    g = torch.Generator()
    g.manual_seed(seed)

    def mk(ds: Optional[Dataset], shuffle: bool) -> Optional[DataLoader]:
        if ds is None:
            return None
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=g if shuffle else None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

    loaders = {
        "train": mk(datasets.get("train"), True),
        "val": mk(datasets.get("val"), False),
        "test_id": mk(datasets.get("test_id"), False),
        "test_cross_env": mk(datasets.get("test_cross_env"), False),
        "test_living_room": mk(datasets.get("test_living_room"), False),
        "test_lecture_room": mk(datasets.get("test_lecture_room"), False),
    }
    return loaders


# =========================================================
# Model registry
# =========================================================
def get_model_config(model_name: str, input_h: int, input_w: int) -> Dict[str, Any]:
    if model_name == "dualmcn_default":
        return {
            "type": "dualmcn",
            "num_classes": 2,
            "in_chans": 1,
            "depths": (1, 1),
            "dims": (32, 64),
            "patch_size": 2,
            "stage1_main_channels": 32,
            "stage1_aux_channels": 32,
            "stage2_main_channels": 64,
            "stage2_aux_channels": 64,
            "use_stem_se": False,
            "use_block_se": True,
            "use_main_proj_bn": True,
            "use_aux_proj_bn": True,
            "use_fusion_proj_bn": True,
            "use_upsample_bn": True,
            "use_wavelet": True,
            "branch_mode": "dual",
            "recon_type": "deconv",
            "aux_variant": "mkconv",
        }

    # Old default: stem SE + no block SE.
    if model_name == "dualmcn_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["use_stem_se"] = True
        cfg["use_block_se"] = False
        return cfg

    # Ablations based on the new default.
    if model_name == "dualmcn_no_wavelet":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["use_wavelet"] = False
        return cfg
    if model_name == "dualmcn_main_only":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["branch_mode"] = "main_only"
        return cfg
    if model_name == "dualmcn_aux_only":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["branch_mode"] = "aux_only"
        return cfg
    if model_name == "dualmcn_recon_interp_1x1":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["recon_type"] = "interp_1x1"
        return cfg
    if model_name == "dualmcn_aux_pw":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["aux_variant"] = "mkconv_pw"
        return cfg

    # Ablations based on the old default.
    if model_name == "dualmcn_no_wavelet_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_stem_se_no_block_se", input_h, input_w)
        cfg["use_wavelet"] = False
        return cfg
    if model_name == "dualmcn_main_only_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_stem_se_no_block_se", input_h, input_w)
        cfg["branch_mode"] = "main_only"
        return cfg
    if model_name == "dualmcn_aux_only_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_stem_se_no_block_se", input_h, input_w)
        cfg["branch_mode"] = "aux_only"
        return cfg
    if model_name == "dualmcn_recon_interp_1x1_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_stem_se_no_block_se", input_h, input_w)
        cfg["recon_type"] = "interp_1x1"
        return cfg
    if model_name == "dualmcn_aux_pw_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_stem_se_no_block_se", input_h, input_w)
        cfg["aux_variant"] = "mkconv_pw"
        return cfg

    # Independent SE ablation.
    if model_name == "dualmcn_no_stem_se_no_block_se":
        cfg = get_model_config("dualmcn_default", input_h, input_w)
        cfg["use_stem_se"] = False
        cfg["use_block_se"] = False
        return cfg

    if model_name == "resnet18":
        return {"type": "resnet18", "num_classes": 2, "in_chans": 1}
    if model_name == "lstm":
        return {
            "type": "lstm",
            "num_classes": 2,
            "input_size": input_w,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.2,
        }
    if model_name == "gru":
        return {
            "type": "gru",
            "num_classes": 2,
            "input_size": input_w,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.2,
        }
    if model_name == "tcn":
        return {
            "type": "tcn",
            "num_classes": 2,
            "input_size": input_w,
            "num_channels": (64, 64, 128, 128, 128, 128),
            "kernel_size": 3,
            "dropout": 0.2,
        }
    if model_name == "vim":
        return {
            "type": "vim",
            "num_classes": 2,
            "img_h": input_h,
            "img_w": input_w,
            "patch_size": 8,
            "stride": 8,
            "depth": 8,
            "embed_dim": 128,
            "d_state": 16,
            "channels": 1,
            "drop_path_rate": 0.05,
        }
    raise ValueError(f"Unknown model_name: {model_name}")

def build_model(model_name: str, device: torch.device, input_h: int, input_w: int):
    cfg = get_model_config(model_name, input_h=input_h, input_w=input_w)
    model_type = cfg.pop("type")
    if model_type == "dualmcn":
        model = DualMCN(**cfg).to(device)
    else:
        registry = load_baseline_registry()
        if model_type not in registry:
            raise ValueError(f"Unknown model type: {model_type}")
        model = registry[model_type](**cfg).to(device)
    return model, get_model_config(model_name, input_h=input_h, input_w=input_w)


# =========================================================
# Eval
# =========================================================
def _extract_xy(batch):
    if len(batch[0]) == 0:
        return None, None
    return batch[0], batch[1]


def evaluate_split(model: nn.Module, loader: Optional[DataLoader], device: torch.device, idx_to_label: Dict[int, str]) -> Optional[Dict[str, Any]]:
    if loader is None:
        return None
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x, y = _extract_xy(batch)
            if x is None:
                continue
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    if not all_labels:
        return None

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    label_indices = sorted(idx_to_label.keys())
    class_names = [idx_to_label[i] for i in label_indices]

    acc = accuracy_score(all_labels_np, all_preds_np)
    macro_f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)

    fall_idx = next((i for i, name in idx_to_label.items() if str(name).lower() == "fall"), None)
    nonfall_idx = next((i for i, name in idx_to_label.items() if str(name).lower() == "nonfall"), None)
    fall_recall = recall_score(all_labels_np, all_preds_np, pos_label=fall_idx, zero_division=0) if fall_idx is not None else None
    nonfall_recall = recall_score(all_labels_np, all_preds_np, pos_label=nonfall_idx, zero_division=0) if nonfall_idx is not None else None

    report_text = classification_report(all_labels_np, all_preds_np, labels=label_indices, target_names=class_names, digits=4, zero_division=0)
    report_structured = classification_report(all_labels_np, all_preds_np, labels=label_indices, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    matrix = confusion_matrix(all_labels_np, all_preds_np, labels=label_indices).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "fall_recall": float(fall_recall) if fall_recall is not None else None,
        "nonfall_recall": float(nonfall_recall) if nonfall_recall is not None else None,
        "report_text": report_text,
        "report_structured": report_structured,
        "confusion_matrix": matrix,
        "labels_order": class_names,
    }


# =========================================================
# Checkpoint policies
# =========================================================
def average_checkpoints(checkpoint_paths: List[str], device: torch.device):
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")
    avg_state = None
    for path in checkpoint_paths:
        state = torch_load(path, device)
        if avg_state is None:
            avg_state = {}
            for k, v in state.items():
                avg_state[k] = v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone()
        else:
            for k, v in state.items():
                if avg_state[k].dtype.is_floating_point:
                    avg_state[k] += v.detach().float()
                else:
                    avg_state[k] = v.detach().clone()
    n = float(len(checkpoint_paths))
    for k, v in avg_state.items():
        if v.dtype.is_floating_point:
            avg_state[k] /= n
    return avg_state


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> bool:
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return False

    was_training = model.training
    model.train()
    old_momenta = {}
    for m in bn_layers:
        m.reset_running_stats()
        old_momenta[m] = m.momentum
        m.momentum = None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, _ = _extract_xy(batch)
        if x is None:
            continue
        model(x.to(device))

    for m in bn_layers:
        m.momentum = old_momenta[m]
    model.train(was_training)
    return True


class CheckpointManager:
    def __init__(self, policy: str, ckpt_dir: str, model_name: str, seed: int, use_ema: bool):
        self.policy = policy
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.seed = seed
        self.use_ema = use_ema
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.raw_best_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}_best_raw.pth")
        self.ema_best_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}_best_ema.pth") if use_ema else None
        self.raw_topk_paths: List[str] = []
        self.ema_topk_paths: List[str] = []

    def update(self, val_macro_f1: float, epoch: int, model: nn.Module, ema_helper: Optional[ModelEMA]):
        is_best = val_macro_f1 > self.best_val_f1
        if not is_best:
            return False

        self.best_val_f1 = float(val_macro_f1)
        self.best_epoch = int(epoch)

        if self.policy == "valbest_raw":
            torch.save(model.state_dict(), self.raw_best_path)
        elif self.policy == "valbest_ema":
            torch.save(model.state_dict(), self.raw_best_path)
            if ema_helper is None:
                raise RuntimeError("checkpoint_policy=valbest_ema but EMA helper is None")
            torch.save(ema_helper.state_dict(), self.ema_best_path)
        elif self.policy == "top5avg_bnrecal":
            raw_ckpt_path = os.path.join(self.ckpt_dir, f"{self.model_name}_seed{self.seed}_valbest_ep{epoch:03d}_raw.pth")
            torch.save(model.state_dict(), raw_ckpt_path)
            self.raw_topk_paths.append(raw_ckpt_path)
            if len(self.raw_topk_paths) > 5:
                old = self.raw_topk_paths.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            if self.use_ema and ema_helper is not None:
                ema_ckpt_path = os.path.join(self.ckpt_dir, f"{self.model_name}_seed{self.seed}_valbest_ep{epoch:03d}_ema.pth")
                torch.save(ema_helper.state_dict(), ema_ckpt_path)
                self.ema_topk_paths.append(ema_ckpt_path)
                if len(self.ema_topk_paths) > 5:
                    old = self.ema_topk_paths.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
        else:
            raise ValueError(f"Unknown checkpoint policy: {self.policy}")
        return True

    def build_eval_model(self, model: nn.Module, train_loader: DataLoader, device: torch.device):
        eval_model = copy.deepcopy(model).to(device)
        eval_checkpoint_type = None
        eval_checkpoint_path = None

        if self.policy == "valbest_raw":
            eval_model.load_state_dict(torch_load(self.raw_best_path, device), strict=True)
            eval_checkpoint_type = "raw_best"
            eval_checkpoint_path = self.raw_best_path
        elif self.policy == "valbest_ema":
            if self.ema_best_path is None:
                raise RuntimeError("EMA best path missing")
            eval_model.load_state_dict(torch_load(self.ema_best_path, device), strict=True)
            eval_checkpoint_type = "ema_best"
            eval_checkpoint_path = self.ema_best_path
        elif self.policy == "top5avg_bnrecal":
            use_ema_eval = self.use_ema and len(self.ema_topk_paths) > 0
            selected_paths = self.ema_topk_paths if use_ema_eval else self.raw_topk_paths
            if not selected_paths:
                raise RuntimeError("No val-best checkpoints were saved, cannot average checkpoints.")
            avg_state = average_checkpoints(selected_paths, device=device)
            avg_ckpt_path = os.path.join(
                self.ckpt_dir,
                f"{self.model_name}_seed{self.seed}_{'ema' if use_ema_eval else 'raw'}_top{len(selected_paths)}avg.pth",
            )
            torch.save(avg_state, avg_ckpt_path)
            eval_model.load_state_dict(avg_state, strict=True)
            bn_recalibrated = recalibrate_batchnorm(eval_model, train_loader, device)
            eval_checkpoint_type = f"{'ema' if use_ema_eval else 'raw'}_top{len(selected_paths)}avg"
            if bn_recalibrated:
                eval_checkpoint_type += "_bnrecal"
            eval_checkpoint_path = avg_ckpt_path
        else:
            raise ValueError(f"Unknown checkpoint policy: {self.policy}")

        return eval_model, eval_checkpoint_type, eval_checkpoint_path


# =========================================================
# Training / running
# =========================================================
def get_lr_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return epoch / warmup_epochs if epoch <= warmup_epochs else 1.0


def run_single_seed(
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    device: torch.device,
    loaders: Dict[str, Optional[DataLoader]],
    idx_to_label: Dict[int, str],
    input_h: int,
    input_w: int,
    run_dir: str,
):
    set_global_seed(seed)

    try:
        model, model_config = build_model(model_name, device, input_h=input_h, input_w=input_w)
    except Exception as e:
        if model_name == "vim":
            warnings.warn(f"[Skip] Failed to build vim for seed={seed}: {e}")
            return None, None
        raise

    params_total, params_trainable = count_parameters(model)
    training_cfg = cfg["training"]
    checkpoint_cfg = cfg["checkpoint"]
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    use_ema = bool(training_cfg.get("use_ema", False))
    ema_helper = None
    if use_ema:
        ema_helper = ModelEMA(model, decay=training_cfg.get("ema_decay", 0.999)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["lr"], weight_decay=training_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])

    manager = CheckpointManager(
        policy=checkpoint_cfg["checkpoint_policy"],
        ckpt_dir=ckpt_dir,
        model_name=model_name,
        seed=seed,
        use_ema=use_ema,
    )

    for epoch in range(1, training_cfg["epochs"] + 1):
        lr_scale = get_lr_scale(epoch, training_cfg["warmup_epochs"])
        for pg in optimizer.param_groups:
            pg["lr"] = training_cfg["lr"] * lr_scale

        model.train()
        for batch in loaders["train"]:
            x, y = _extract_xy(batch)
            if x is None:
                continue
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_helper is not None:
                ema_helper.update(model)

        if epoch > training_cfg["warmup_epochs"]:
            scheduler.step()

        val_metrics = evaluate_split(model, loaders["val"], device, idx_to_label)
        if val_metrics is None:
            raise RuntimeError(f"Validation produced no metrics for model={model_name}, seed={seed}")
        is_best = manager.update(val_metrics["macro_f1"], epoch, model, ema_helper)
        print(
            f"[{model_name} seed={seed}] Ep {epoch:03d} | "
            f"Val Acc={val_metrics['accuracy']*100:.2f}% | "
            f"Val F1={val_metrics['macro_f1']*100:.2f}%"
            + ("  <-- Best" if is_best else "")
        )

    eval_model, eval_checkpoint_type, eval_checkpoint_path = manager.build_eval_model(model, loaders["train"], device)
    row = {
        "model_name": model_name,
        "seed": int(seed),
        "params_M_total": params_total,
        "params_M_trainable": params_trainable,
        "best_epoch": int(manager.best_epoch),
        "best_val_macro_f1": float(manager.best_val_f1),
        "best_checkpoint_path": eval_checkpoint_path,
        "eval_checkpoint_type": eval_checkpoint_type,
        "use_ema": 1.0 if use_ema else 0.0,
        "ema_decay": float(training_cfg.get("ema_decay", 0.999)) if use_ema else None,
    }

    # Always evaluate val for record consistency.
    val_eval = evaluate_split(eval_model, loaders["val"], device, idx_to_label)
    if val_eval is not None:
        row.update({
            "val_accuracy": float(val_eval["accuracy"]),
            "val_macro_f1": float(val_eval["macro_f1"]),
            "val_fall_recall": float(val_eval["fall_recall"]) if val_eval["fall_recall"] is not None else None,
            "val_nonfall_recall": float(val_eval["nonfall_recall"]) if val_eval["nonfall_recall"] is not None else None,
            "val_confusion_matrix": val_eval["confusion_matrix"],
            "val_report_structured": val_eval["report_structured"],
            "val_report_text": val_eval["report_text"],
            "val_labels_order": val_eval["labels_order"],
        })

    for split_name in ["test_id", "test_cross_env", "test_living_room", "test_lecture_room"]:
        metrics = evaluate_split(eval_model, loaders.get(split_name), device, idx_to_label)
        if metrics is None:
            continue
        row.update({
            f"{split_name}_accuracy": float(metrics["accuracy"]),
            f"{split_name}_macro_f1": float(metrics["macro_f1"]),
            f"{split_name}_fall_recall": float(metrics["fall_recall"]) if metrics["fall_recall"] is not None else None,
            f"{split_name}_nonfall_recall": float(metrics["nonfall_recall"]) if metrics["nonfall_recall"] is not None else None,
            f"{split_name}_confusion_matrix": metrics["confusion_matrix"],
            f"{split_name}_report_structured": metrics["report_structured"],
            f"{split_name}_report_text": metrics["report_text"],
            f"{split_name}_labels_order": metrics["labels_order"],
        })

    return row, model_config


# =========================================================
# Summaries
# =========================================================
def build_dataset_summary(dataset_name: str, datasets: Dict[str, Any]) -> Dict[str, Any]:
    def summarize(ds: Optional[Dataset], name: str):
        if ds is None:
            return None
        entry = {
            "name": name,
            "num_samples": int(len(ds)),
        }
        if hasattr(ds, "num_classes"):
            entry["num_classes"] = int(getattr(ds, "num_classes"))
        return entry

    if dataset_name == "enetfall":
        out = {
            "train": summarize(datasets.get("train"), "train"),
            "val": summarize(datasets.get("val"), "val"),
            "test_living_room": summarize(datasets.get("test_living_room"), "test_living_room"),
            "test_lecture_room": summarize(datasets.get("test_lecture_room"), "test_lecture_room"),
        }
        if "env_stats" in datasets:
            out["env_stats"] = datasets["env_stats"]
        return out

    return {
        "train": summarize(datasets.get("train"), "train"),
        "val": summarize(datasets.get("val"), "val"),
        "test_id": summarize(datasets.get("test_id"), "test_id"),
        "test_cross_env": summarize(datasets.get("test_cross_env"), "test_cross_env"),
    }


def build_summary_rows(models_payload: Dict[str, Any], report_splits: List[str], model_order: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for model_name in model_order:
        if model_name not in models_payload:
            continue
        agg = models_payload[model_name]["aggregate"]
        row = {
            "model": model_name,
            "params_M_total": format_mean_std(agg.get("params_M_total_mean"), agg.get("params_M_total_std"), digits=4),
            "best_epoch": format_mean_std(agg.get("best_epoch_mean"), agg.get("best_epoch_std"), digits=2),
            "best_val_macro_f1": format_mean_std(agg.get("best_val_macro_f1_mean"), agg.get("best_val_macro_f1_std"), digits=4),
            "eval_checkpoint_type": models_payload[model_name].get("eval_checkpoint_type", "-"),
        }
        for split_name in report_splits:
            row[f"{split_name}_accuracy"] = format_mean_std(agg.get(f"{split_name}_accuracy_mean"), agg.get(f"{split_name}_accuracy_std"), digits=4)
            row[f"{split_name}_macro_f1"] = format_mean_std(agg.get(f"{split_name}_macro_f1_mean"), agg.get(f"{split_name}_macro_f1_std"), digits=4)
            row[f"{split_name}_fall_recall"] = format_mean_std(agg.get(f"{split_name}_fall_recall_mean"), agg.get(f"{split_name}_fall_recall_std"), digits=4)
            row[f"{split_name}_nonfall_recall"] = format_mean_std(agg.get(f"{split_name}_nonfall_recall_mean"), agg.get(f"{split_name}_nonfall_recall_std"), digits=4)
        rows.append(row)
    return rows


def save_csv(rows: List[Dict[str, Any]], path: str):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_md(rows: List[Dict[str, Any]], path: str, title: str):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\nNo rows.\n")
        return
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


# =========================================================
# Experiment runner
# =========================================================
def run_experiment_group(cfg: Dict[str, Any], exp_name: str, exp_cfg: Dict[str, Any], device: torch.device):
    if not exp_cfg.get("enabled", True):
        print(f"[Skip] experiment '{exp_name}' disabled by config")
        return None

    output_root = cfg["output"]["output_root"]
    run_dir = os.path.join(output_root, exp_name)
    ensure_dir(run_dir)

    results = {
        "config_path": None,
        "dataset": cfg["dataset"],
        "training": cfg["training"],
        "checkpoint": cfg["checkpoint"],
        "experiment_name": exp_name,
        "use_augmentation": exp_cfg["use_augmentation"],
        "requested_models": exp_cfg["models"],
        "models": {},
        "skipped_models": [],
    }

    for model_name in exp_cfg["models"]:
        print("\n" + "#" * 120)
        print(f"[Experiment={exp_name}] [Model] {model_name}")
        print("#" * 120)
        per_seed = []
        model_config = None
        eval_checkpoint_type_set = set()

        for seed in cfg["training"]["seeds"]:
            print("\n" + "=" * 100)
            print(f"[Run] experiment={exp_name} | model={model_name} | seed={seed}")
            print("=" * 100)
            datasets = build_datasets(cfg, use_augmentation=exp_cfg["use_augmentation"], seed=seed)
            if seed == cfg["training"]["seeds"][0]:
                results["dataset_summary"] = build_dataset_summary(cfg["dataset"]["name"], datasets)
            loaders = build_loaders(cfg, datasets, seed=seed)
            row, model_config = run_single_seed(
                cfg=cfg,
                model_name=model_name,
                seed=seed,
                device=device,
                loaders=loaders,
                idx_to_label=datasets["idx_to_label"],
                input_h=datasets["input_h"],
                input_w=datasets["input_w"],
                run_dir=run_dir,
            )
            if row is None:
                if model_name not in results["skipped_models"]:
                    results["skipped_models"].append(model_name)
                per_seed = []
                break
            eval_checkpoint_type_set.add(row["eval_checkpoint_type"])
            per_seed.append(row)
            msg = (
                f"[Summary] {model_name} | seed={seed} | best_val_f1={row['best_val_macro_f1']:.4f} | "
                f"best_epoch={row['best_epoch']} | eval_ckpt={row['eval_checkpoint_type']}"
            )
            for split_name in cfg["dataset"]["report_splits"]:
                if f"{split_name}_macro_f1" in row:
                    msg += f" | {split_name}_f1={row[f'{split_name}_macro_f1']:.4f} | {split_name}_acc={row[f'{split_name}_accuracy']:.4f}"
            print(msg)

        if not per_seed:
            continue
        aggregate = aggregate_numeric_dicts(per_seed)
        results["models"][model_name] = {
            "model_config": model_config,
            "per_seed": per_seed,
            "aggregate": aggregate,
            "eval_checkpoint_type": sorted(eval_checkpoint_type_set)[0] if len(eval_checkpoint_type_set) == 1 else ",".join(sorted(eval_checkpoint_type_set)),
        }

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compare vs ablation tables.
    compare_models = [m for m in cfg["models"]["models_main"] if m in results["models"]]
    ablation_models = [m for m in cfg["models"]["models_ablation"] if m in results["models"]]
    report_splits = cfg["dataset"]["report_splits"]

    compare_rows = build_summary_rows(results["models"], report_splits=report_splits, model_order=compare_models)
    ablation_rows = build_summary_rows(results["models"], report_splits=report_splits, model_order=ablation_models)

    if cfg["output"].get("save_csv", True):
        save_csv(compare_rows, os.path.join(run_dir, "summary_compare.csv"))
        save_csv(ablation_rows, os.path.join(run_dir, "summary_ablation.csv"))
    if cfg["output"].get("save_md", True):
        save_md(compare_rows, os.path.join(run_dir, "summary_compare.md"), title=f"{exp_name} - Compare")
        save_md(ablation_rows, os.path.join(run_dir, "summary_ablation.md"), title=f"{exp_name} - Ablation")

    return results


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Unified run for CSI-Bench / ENetFall / ourdata")
    parser.add_argument("--config", type=str, required=True, help="Path to a single-dataset JSON config")
    args = parser.parse_args()

    cfg = load_json(args.config)
    validate_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg["output"]["output_root"])

    # Save resolved config snapshot.
    snapshot_path = os.path.join(cfg["output"]["output_root"], "resolved_config.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("[Config]")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"[Device] {device}")

    for exp_name, exp_cfg in cfg["experiments"].items():
        run_experiment_group(cfg, exp_name, exp_cfg, device=device)


if __name__ == "__main__":
    main()
