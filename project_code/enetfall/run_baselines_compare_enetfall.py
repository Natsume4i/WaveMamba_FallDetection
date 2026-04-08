import os
import json
import copy
import random
import argparse
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from enetfall_dataset import ENetFallDataset, robust_collate_fn
from wavemamba import WaveMamba
from baseline_models import ResNet18Baseline, BiLSTMBaseline


TRAIN_ENVS = [
    "dataset_meeting_room.mat",
    "dataset_home_lab(L).mat",
    "dataset_home_lab(R).mat",
]

LIVING_ROOM_ENV = "dataset_living_room.mat"
LECTURE_ROOM_ENV = "dataset_lecture_room.mat"
CLASS_NAMES = ["Nonfall", "Fall"]


class ENetFallCacheSubset(Dataset):
    def __init__(self, base_dataset, indices, augment=False):
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
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_sizes = []
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


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return float(total), float(trainable)


def format_mean_std(m, s, digits=4):
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def safe_format_mean_std(m, s, digits=4):
    if m is None:
        return "-"
    return format_mean_std(float(m), float(0.0 if s is None else s), digits=digits)


def aggregate_numeric_dicts(rows):
    if not rows:
        return {}
    out = {}
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    for k in numeric_keys:
        vals = [float(x[k]) for x in rows if k in x and isinstance(x[k], (int, float))]
        if len(vals) == 0:
            continue
        out[f"{k}_mean"] = float(mean(vals))
        out[f"{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def build_dataset_summary(train_ds, val_ds, test_living_room_ds=None, test_lecture_room_ds=None):
    def summarize(ds, name):
        if ds is None:
            return None
        return {
            "name": name,
            "num_samples": int(len(ds)),
            "num_classes": 2,
        }

    return {
        "train": summarize(train_ds, "train"),
        "val": summarize(val_ds, "val"),
        "test_living_room": summarize(test_living_room_ds, "test_living_room"),
        "test_lecture_room": summarize(test_lecture_room_ds, "test_lecture_room"),
    }


def evaluate_split(model, loader, device, idx_to_label):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch[0]) == 0:
                continue
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    label_indices = sorted(idx_to_label.keys())
    class_names = [idx_to_label[i] for i in label_indices]

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    fall_idx = next((i for i, name in idx_to_label.items() if name.lower() == "fall"), None)
    nonfall_idx = next((i for i, name in idx_to_label.items() if name.lower() == "nonfall"), None)

    fall_recall = recall_score(all_labels, all_preds, pos_label=fall_idx, zero_division=0) if fall_idx is not None else None
    nonfall_recall = recall_score(all_labels, all_preds, pos_label=nonfall_idx, zero_division=0) if nonfall_idx is not None else None

    report_text = classification_report(
        all_labels, all_preds, labels=label_indices, target_names=class_names, digits=4, zero_division=0
    )
    report_structured = classification_report(
        all_labels, all_preds, labels=label_indices, target_names=class_names, digits=4, zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(all_labels, all_preds, labels=label_indices).tolist()

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


def get_model_config(model_name: str):
    if model_name == "wavemamba_default":
        return {
            "type": "wavemamba",
            "num_classes": 2,
            "in_chans": 1,
            "depths": (1, 1),
            "dims": (32, 64),
            "patch_size": 2,
            "stage1_main_channels": 32,
            "stage1_aux_channels": 32,
            "stage2_main_channels": 64,
            "stage2_aux_channels": 64,
            "use_stem_se": True,
            "use_block_se": False,
            "use_main_proj_bn": True,
            "use_aux_proj_bn": True,
            "use_fusion_proj_bn": True,
            "use_upsample_bn": True,
            "use_wavelet": True,
            "branch_mode": "dual",
            "recon_type": "deconv",
            "aux_variant": "mkconv",
        }
    if model_name == "wavemamba_interp_1x1":
        return {
            "type": "wavemamba",
            "num_classes": 2,
            "in_chans": 1,
            "depths": (1, 1),
            "dims": (32, 64),
            "patch_size": 2,
            "stage1_main_channels": 32,
            "stage1_aux_channels": 32,
            "stage2_main_channels": 64,
            "stage2_aux_channels": 64,
            "use_stem_se": True,
            "use_block_se": False,
            "use_main_proj_bn": True,
            "use_aux_proj_bn": True,
            "use_fusion_proj_bn": True,
            "use_upsample_bn": True,
            "use_wavelet": True,
            "branch_mode": "dual",
            "recon_type": "interp_1x1",
            "aux_variant": "mkconv",
        }
    if model_name == "resnet18":
        return {"type": "resnet18", "num_classes": 2, "in_chans": 1}
    if model_name == "lstm":
        return {
            "type": "lstm",
            "num_classes": 2,
            "input_size": 90,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.2,
        }
    raise ValueError(f"Unknown model_name: {model_name}")


def build_model(model_name: str, device):
    cfg = get_model_config(model_name)
    model_type = cfg.pop("type")

    if model_type == "wavemamba":
        model = WaveMamba(**cfg).to(device)
    elif model_type == "resnet18":
        model = ResNet18Baseline(**cfg).to(device)
    elif model_type == "lstm":
        model = BiLSTMBaseline(**cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, get_model_config(model_name)


def stratified_split_indices_from_cache(base_dataset, val_ratio, seed):
    labels = [label for _, label in base_dataset.cache]
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    return train_idx.tolist(), val_idx.tolist()


def build_datasets(args, seed):
    train_subsets = []
    val_subsets = []
    env_stats = {}

    for env_name in TRAIN_ENVS:
        base_env_ds = ENetFallDataset(
            data_root=args.data_root,
            file_list=[env_name],
            augment=True,
        )
        train_idx, val_idx = stratified_split_indices_from_cache(base_env_ds, args.val_ratio, seed)

        train_subsets.append(ENetFallCacheSubset(base_env_ds, train_idx, augment=True))
        val_subsets.append(ENetFallCacheSubset(base_env_ds, val_idx, augment=False))

        labels = [label for _, label in base_env_ds.cache]
        env_stats[env_name] = {
            "total": len(labels),
            "fall": int(np.sum(np.array(labels) == 1)),
            "nonfall": int(np.sum(np.array(labels) == 0)),
            "train": len(train_idx),
            "val": len(val_idx),
        }

    train_ds = ConcatSimpleDataset(train_subsets)
    val_ds = ConcatSimpleDataset(val_subsets)

    test_living_room_ds = ENetFallDataset(
        data_root=args.data_root,
        file_list=[LIVING_ROOM_ENV],
        augment=False,
    )
    test_lecture_room_ds = ENetFallDataset(
        data_root=args.data_root,
        file_list=[LECTURE_ROOM_ENV],
        augment=False,
    )

    idx_to_label = {0: "Nonfall", 1: "Fall"}
    return train_ds, val_ds, test_living_room_ds, test_lecture_room_ds, idx_to_label, env_stats


def build_loaders_from_datasets(args, seed, train_ds, val_ds, test_living_room_ds, test_lecture_room_ds):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=robust_collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=robust_collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    test_living_room_loader = DataLoader(
        test_living_room_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=robust_collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    test_lecture_room_loader = DataLoader(
        test_lecture_room_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=robust_collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_living_room_loader, test_lecture_room_loader


def get_lr_scale(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def run_single_seed(model_name, seed, args, device, train_loader, val_loader, test_living_room_loader, test_lecture_room_loader, idx_to_label):
    set_global_seed(seed)
    model, model_config = build_model(model_name, device)
    params_total, params_trainable = count_parameters(model)

    ema_helper = None
    if args.use_ema:
        ema_helper = ModelEMA(model, decay=args.ema_decay)
        ema_helper.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_epoch = -1
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    raw_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}_best_raw.pth")
    ema_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}_best_ema.pth") if args.use_ema else None

    for epoch in range(1, args.epochs + 1):
        lr_scale = get_lr_scale(epoch, args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_scale

        model.train()
        for batch in train_loader:
            if len(batch[0]) == 0:
                continue
            x, y = batch[0].to(device), batch[1].to(device)

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

        if epoch > args.warmup_epochs:
            scheduler.step()

        val_metrics = evaluate_split(model, val_loader, device, idx_to_label)
        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), raw_ckpt_path)
            if ema_helper is not None:
                torch.save(ema_helper.state_dict(), ema_ckpt_path)

        print(
            f"[{model_name} seed={seed}] "
            f"Ep {epoch:03d} | "
            f"Val Acc={val_metrics['accuracy']*100:.2f}% | "
            f"Val F1={val_metrics['macro_f1']*100:.2f}%"
            + ("  <-- Best" if is_best else "")
        )

    eval_model = model
    eval_checkpoint_type = "raw_best"
    eval_checkpoint_path = raw_ckpt_path

    if args.use_ema:
        model.load_state_dict(torch.load(raw_ckpt_path, map_location=device))
        ema_eval = copy.deepcopy(model).to(device)
        ema_eval.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
        eval_model = ema_eval
        eval_checkpoint_type = "ema_best"
        eval_checkpoint_path = ema_ckpt_path
    else:
        model.load_state_dict(torch.load(raw_ckpt_path, map_location=device))
        eval_model = model

    best_val_metrics = evaluate_split(eval_model, val_loader, device, idx_to_label)
    test_train_env_metrics = evaluate_split(eval_model, val_loader, device, idx_to_label)
    test_living_room_metrics = evaluate_split(eval_model, test_living_room_loader, device, idx_to_label)
    test_lecture_room_metrics = evaluate_split(eval_model, test_lecture_room_loader, device, idx_to_label)

    row = {
        "model_name": model_name,
        "seed": int(seed),
        "params_M_total": params_total,
        "params_M_trainable": params_trainable,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "best_checkpoint_path": eval_checkpoint_path,
        "eval_checkpoint_type": eval_checkpoint_type,
        "use_ema": 1.0 if args.use_ema else 0.0,
        "ema_decay": float(args.ema_decay) if args.use_ema else None,

        "test_train_env_accuracy": float(test_train_env_metrics["accuracy"]),
        "test_train_env_macro_f1": float(test_train_env_metrics["macro_f1"]),
        "test_train_env_fall_recall": float(test_train_env_metrics["fall_recall"]) if test_train_env_metrics["fall_recall"] is not None else None,
        "test_train_env_nonfall_recall": float(test_train_env_metrics["nonfall_recall"]) if test_train_env_metrics["nonfall_recall"] is not None else None,
        "test_train_env_confusion_matrix": test_train_env_metrics["confusion_matrix"],
        "test_train_env_report_structured": test_train_env_metrics["report_structured"],
        "test_train_env_report_text": test_train_env_metrics["report_text"],
        "test_train_env_labels_order": test_train_env_metrics["labels_order"],

        "test_living_room_accuracy": float(test_living_room_metrics["accuracy"]),
        "test_living_room_macro_f1": float(test_living_room_metrics["macro_f1"]),
        "test_living_room_fall_recall": float(test_living_room_metrics["fall_recall"]) if test_living_room_metrics["fall_recall"] is not None else None,
        "test_living_room_nonfall_recall": float(test_living_room_metrics["nonfall_recall"]) if test_living_room_metrics["nonfall_recall"] is not None else None,
        "test_living_room_confusion_matrix": test_living_room_metrics["confusion_matrix"],
        "test_living_room_report_structured": test_living_room_metrics["report_structured"],
        "test_living_room_report_text": test_living_room_metrics["report_text"],
        "test_living_room_labels_order": test_living_room_metrics["labels_order"],

        "test_lecture_room_accuracy": float(test_lecture_room_metrics["accuracy"]),
        "test_lecture_room_macro_f1": float(test_lecture_room_metrics["macro_f1"]),
        "test_lecture_room_fall_recall": float(test_lecture_room_metrics["fall_recall"]) if test_lecture_room_metrics["fall_recall"] is not None else None,
        "test_lecture_room_nonfall_recall": float(test_lecture_room_metrics["nonfall_recall"]) if test_lecture_room_metrics["nonfall_recall"] is not None else None,
        "test_lecture_room_confusion_matrix": test_lecture_room_metrics["confusion_matrix"],
        "test_lecture_room_report_structured": test_lecture_room_metrics["report_structured"],
        "test_lecture_room_report_text": test_lecture_room_metrics["report_text"],
        "test_lecture_room_labels_order": test_lecture_room_metrics["labels_order"],
    }

    return row, model_config


def build_summary_rows(results_dict):
    rows = []
    for model_name, payload in results_dict.items():
        agg = payload["aggregate"]
        rows.append({
            "model": model_name,
            "params_M_total_mean": agg.get("params_M_total_mean"),
            "params_M_total_std": agg.get("params_M_total_std"),
            "best_epoch_mean": agg.get("best_epoch_mean"),
            "best_epoch_std": agg.get("best_epoch_std"),
            "best_val_macro_f1_mean": agg.get("best_val_macro_f1_mean"),
            "best_val_macro_f1_std": agg.get("best_val_macro_f1_std"),
            "test_train_env_accuracy_mean": agg.get("test_train_env_accuracy_mean"),
            "test_train_env_accuracy_std": agg.get("test_train_env_accuracy_std"),
            "test_train_env_macro_f1_mean": agg.get("test_train_env_macro_f1_mean"),
            "test_train_env_macro_f1_std": agg.get("test_train_env_macro_f1_std"),
            "test_train_env_fall_recall_mean": agg.get("test_train_env_fall_recall_mean"),
            "test_train_env_fall_recall_std": agg.get("test_train_env_fall_recall_std"),
            "test_train_env_nonfall_recall_mean": agg.get("test_train_env_nonfall_recall_mean"),
            "test_train_env_nonfall_recall_std": agg.get("test_train_env_nonfall_recall_std"),
            "test_living_room_accuracy_mean": agg.get("test_living_room_accuracy_mean"),
            "test_living_room_accuracy_std": agg.get("test_living_room_accuracy_std"),
            "test_living_room_macro_f1_mean": agg.get("test_living_room_macro_f1_mean"),
            "test_living_room_macro_f1_std": agg.get("test_living_room_macro_f1_std"),
            "test_living_room_fall_recall_mean": agg.get("test_living_room_fall_recall_mean"),
            "test_living_room_fall_recall_std": agg.get("test_living_room_fall_recall_std"),
            "test_living_room_nonfall_recall_mean": agg.get("test_living_room_nonfall_recall_mean"),
            "test_living_room_nonfall_recall_std": agg.get("test_living_room_nonfall_recall_std"),
            "test_lecture_room_accuracy_mean": agg.get("test_lecture_room_accuracy_mean"),
            "test_lecture_room_accuracy_std": agg.get("test_lecture_room_accuracy_std"),
            "test_lecture_room_macro_f1_mean": agg.get("test_lecture_room_macro_f1_mean"),
            "test_lecture_room_macro_f1_std": agg.get("test_lecture_room_macro_f1_std"),
            "test_lecture_room_fall_recall_mean": agg.get("test_lecture_room_fall_recall_mean"),
            "test_lecture_room_fall_recall_std": agg.get("test_lecture_room_fall_recall_std"),
            "test_lecture_room_nonfall_recall_mean": agg.get("test_lecture_room_nonfall_recall_mean"),
            "test_lecture_room_nonfall_recall_std": agg.get("test_lecture_room_nonfall_recall_std"),
        })

    rows = sorted(rows, key=lambda x: x.get("test_living_room_macro_f1_mean", -1.0) if x.get("test_living_room_macro_f1_mean") is not None else -1.0, reverse=True)
    return rows


def save_csv(summary_rows, out_path):
    pd.DataFrame(summary_rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_md(summary_rows, out_path, use_ema=False, ema_decay=None):
    lines = []
    lines.append(f"EMA enabled: {use_ema}")
    if use_ema:
        lines.append(f"EMA decay: {ema_decay}")
    lines.append("")
    lines.append("| Model | Params(M) | Best Val F1 | TrainEnv Acc | TrainEnv F1 | TrainEnv Fall R | TrainEnv Nonfall R | LivingRoom Acc | LivingRoom F1 | LivingRoom Fall R | LivingRoom Nonfall R | LectureRoom Acc | LectureRoom F1 | LectureRoom Fall R | LectureRoom Nonfall R | Best Epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['model']} "
            f"| {safe_format_mean_std(row['params_M_total_mean'], row['params_M_total_std'])} "
            f"| {safe_format_mean_std(row['best_val_macro_f1_mean'], row['best_val_macro_f1_std'])} "
            f"| {safe_format_mean_std(row['test_train_env_accuracy_mean'], row['test_train_env_accuracy_std'])} "
            f"| {safe_format_mean_std(row['test_train_env_macro_f1_mean'], row['test_train_env_macro_f1_std'])} "
            f"| {safe_format_mean_std(row['test_train_env_fall_recall_mean'], row['test_train_env_fall_recall_std'])} "
            f"| {safe_format_mean_std(row['test_train_env_nonfall_recall_mean'], row['test_train_env_nonfall_recall_std'])} "
            f"| {safe_format_mean_std(row['test_living_room_accuracy_mean'], row['test_living_room_accuracy_std'])} "
            f"| {safe_format_mean_std(row['test_living_room_macro_f1_mean'], row['test_living_room_macro_f1_std'])} "
            f"| {safe_format_mean_std(row['test_living_room_fall_recall_mean'], row['test_living_room_fall_recall_std'])} "
            f"| {safe_format_mean_std(row['test_living_room_nonfall_recall_mean'], row['test_living_room_nonfall_recall_std'])} "
            f"| {safe_format_mean_std(row['test_lecture_room_accuracy_mean'], row['test_lecture_room_accuracy_std'])} "
            f"| {safe_format_mean_std(row['test_lecture_room_macro_f1_mean'], row['test_lecture_room_macro_f1_std'])} "
            f"| {safe_format_mean_std(row['test_lecture_room_fall_recall_mean'], row['test_lecture_room_fall_recall_std'])} "
            f"| {safe_format_mean_std(row['test_lecture_room_nonfall_recall_mean'], row['test_lecture_room_nonfall_recall_std'])} "
            f"| {safe_format_mean_std(row['best_epoch_mean'], row['best_epoch_std'], digits=2)} |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser("ENetFall baseline / final-candidate comparison runner")
    parser.add_argument("--data_root", type=str, default="/mnt/F/natsume/enetfall/data")
    parser.add_argument("--save_dir", type=str, default="./outputs_model_compare_enetfall")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 3407, 777])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--models", type=str, nargs="+", default=["wavemamba_default", "wavemamba_interp_1x1", "resnet18", "lstm"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "data_root": args.data_root,
        "train_envs": TRAIN_ENVS,
        "test_living_room": LIVING_ROOM_ENV,
        "test_lecture_room": LECTURE_ROOM_ENV,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "seeds": args.seeds,
        "models": args.models,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay if args.use_ema else None,
        "best_checkpoint_criterion": "raw_val_macro_f1",
        "final_eval_checkpoint_type": "ema_best" if args.use_ema else "raw_best",
        "final_report_metrics": [
            "test_train_env_macro_f1", "test_train_env_accuracy",
            "test_living_room_macro_f1", "test_living_room_accuracy",
            "test_lecture_room_macro_f1", "test_lecture_room_accuracy",
            "test_train_env_fall_recall", "test_train_env_nonfall_recall",
            "test_living_room_fall_recall", "test_living_room_nonfall_recall",
            "test_lecture_room_fall_recall", "test_lecture_room_nonfall_recall",
        ],
        "model_files": {
            "wavemamba": "wavemamba.py",
            "baselines": "baseline_models.py",
        },
    }

    print("\n[Config]")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    sample_seed = args.seeds[0]
    train_ds, val_ds, test_living_room_ds, test_lecture_room_ds, idx_to_label, env_stats = build_datasets(args, sample_seed)
    dataset_summary = build_dataset_summary(train_ds, val_ds, test_living_room_ds, test_lecture_room_ds)
    dataset_summary["env_stats"] = env_stats
    print("\n[Dataset Summary]")
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

    final_results = {"config": config, "dataset_summary": dataset_summary, "models": {}}

    for model_name in args.models:
        print("\n" + "#" * 120)
        print(f"[Model] {model_name}")
        print("#" * 120)

        per_seed = []
        model_config = None

        for seed in args.seeds:
            print("\n" + "=" * 100)
            print(f"[Run] model={model_name} | seed={seed}")
            print("=" * 100)

            train_ds, val_ds, test_living_room_ds, test_lecture_room_ds, idx_to_label, env_stats = build_datasets(args, seed)
            train_loader, val_loader, test_living_room_loader, test_lecture_room_loader = build_loaders_from_datasets(
                args, seed, train_ds, val_ds, test_living_room_ds, test_lecture_room_ds
            )

            row, model_config = run_single_seed(
                model_name=model_name,
                seed=seed,
                args=args,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_living_room_loader=test_living_room_loader,
                test_lecture_room_loader=test_lecture_room_loader,
                idx_to_label=idx_to_label,
            )
            row["env_stats"] = env_stats
            per_seed.append(row)

            msg = (
                f"[Summary] {model_name} | seed={seed} | "
                f"best_val_f1={row['best_val_macro_f1']:.4f} | "
                f"best_epoch={row['best_epoch']} | "
                f"eval_ckpt={row['eval_checkpoint_type']} | "
                f"trainenv_f1={row['test_train_env_macro_f1']:.4f} | trainenv_acc={row['test_train_env_accuracy']:.4f} | "
                f"livingroom_f1={row['test_living_room_macro_f1']:.4f} | livingroom_acc={row['test_living_room_accuracy']:.4f} | "
                f"lectureroom_f1={row['test_lecture_room_macro_f1']:.4f} | lectureroom_acc={row['test_lecture_room_accuracy']:.4f}"
            )
            print(msg)

        aggregate = aggregate_numeric_dicts(per_seed)
        final_results["models"][model_name] = {
            "model_config": model_config,
            "per_seed": per_seed,
            "aggregate": aggregate,
        }

        with open(os.path.join(args.save_dir, "compare_results.json"), "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    summary_rows = build_summary_rows(final_results["models"])
    final_results["ranking_by_test_living_room_macro_f1"] = [row["model"] for row in summary_rows]

    json_path = os.path.join(args.save_dir, "compare_results.json")
    csv_path = os.path.join(args.save_dir, "compare_summary.csv")
    md_path = os.path.join(args.save_dir, "compare_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    save_csv(summary_rows, csv_path)
    save_md(summary_rows, md_path, use_ema=args.use_ema, ema_decay=args.ema_decay if args.use_ema else None)

    print(f"\n[Saved] JSON -> {json_path}")
    print(f"[Saved] CSV  -> {csv_path}")
    print(f"[Saved] MD   -> {md_path}")


if __name__ == "__main__":
    main()
