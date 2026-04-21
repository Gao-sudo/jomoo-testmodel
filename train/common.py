from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


CLASS_NAMES = [
    "九牧增压花洒",
    "九牧增压花洒套装",
    "九牧大冲力喷枪角阀",
    "九牧安全快开",
    "九牧安全角阀",
    "九牧百搭下水",
    "九牧百搭下水（软袋）",
    "九牧轻音盖板",
    "九牧防断裂淋浴软管",
    "九牧防漏水件",
    "九牧防爆编织软管",
    "九牧防臭下水管",
    "九牧防臭地漏",
    "九牧健康编织软管",
]
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DATA_YAML = Path("yolo.yaml")
DEFAULT_PROJECT = "runs/jomoo"
DEFAULT_OUTPUT_ROOT = Path("outputs")

# 训练基础参数
DEFAULT_EPOCHS = 200          # 保留，但会靠 patience 自动停
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 4             # 保持不变
DEFAULT_WORKERS = 4
DEFAULT_SEED = 42
DEFAULT_DEVICE = "0"

# 早停
DEFAULT_PATIENCE = 30         # 从 50 → 30，更严格

# 优化器 & 学习率
DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_COS_LR = True         # 打开余弦退火，收敛更稳
DEFAULT_LR0 = 0.001           # 从 0.005 → 0.001（大幅降低，解决框不准）
DEFAULT_WEIGHT_DECAY = 0.001  # 从 0.0005 → 0.001，防过拟合

# 增强策略（大幅强化，解决遮挡/漏检）
DEFAULT_CLOSE_MOSAIC = 20     # 晚关 mosaic，小目标更稳

DEFAULT_HSV_H = 0.015
DEFAULT_HSV_S = 0.6           # 增强色彩
DEFAULT_HSV_V = 0.4           # 增强亮度
DEFAULT_DEGREES = 10          # 允许小角度旋转
DEFAULT_SCALE = 0.5           # 增强缩放
DEFAULT_TRANSLATE = 0.15
DEFAULT_MOSAIC = 0.8          # 强化多图拼接
DEFAULT_MIXUP = 0.15
DEFAULT_FLIPLR = 0.5

MODEL_TRAIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "yolov8s": {"epochs": 250, "patience": 50},
    "yolov9c": {
        "epochs": 200,
        "patience": 50,
        "optimizer": "auto",
        "cos_lr": False,
        "lr0": 0.005,
        "mixup": 0.1,
        "close_mosaic": 10,
    },
    "yolov11s": {"epochs": 200, "patience": 40},
}



@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    weights: str
    run_name: str


@dataclass(frozen=True)
class TrainArtifacts:
    run_dir: Path
    artifact_root: Path
    weights_dir: Path
    visualizations_dir: Path
    logs_dir: Path
    meta_dir: Path


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="数据集根目录，默认是项目下的 data 文件夹。",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="数据集配置文件，默认是项目根目录下的 yolo.yaml。",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数。")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="输入图片尺寸。")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch size，默认 4。")
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="训练设备，例如 0、0,1、cpu；默认 0。",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="数据加载线程数。")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="早停容忍轮数。")
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULT_OPTIMIZER,
        help="优化器名称，默认 auto。",
    )
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COS_LR,
        help="是否启用余弦学习率调度。",
    )
    parser.add_argument("--lr0", type=float, default=DEFAULT_LR0, help="初始学习率。")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="权重衰减。",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=DEFAULT_CLOSE_MOSAIC,
        help="最后多少个 epoch 关闭 mosaic。",
    )
    parser.add_argument("--mixup", type=float, default=DEFAULT_MIXUP, help="mixup 增强概率。")
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="Ultralytics 原始训练输出目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="归档后的标准输出根目录（含 weights/visualizations/logs/meta）。",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="归档目录标识；不传则使用时间戳。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子。")


def apply_model_defaults(parser: argparse.ArgumentParser, model_name: str) -> None:
    defaults = MODEL_TRAIN_DEFAULTS.get(model_name)
    if defaults is None:
        return
    parser.set_defaults(**defaults)


def resolve_model_defaults(model_name: str) -> dict[str, Any]:
    return dict(MODEL_TRAIN_DEFAULTS.get(model_name, {}))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def validate_data_layout(data_root: Path) -> None:
    images_train = data_root / "images" / "train"
    images_val = data_root / "images" / "val"
    labels_train = data_root / "labels" / "train"
    labels_val = data_root / "labels" / "val"

    missing = [p for p in [images_train, images_val, labels_train, labels_val] if not p.exists()]
    if missing:
        print("[警告] 以下路径不存在，请确认数据集已按 YOLO 格式整理：")
        for item in missing:
            print(f"  - {item}")
        print("[提示] 如果你只是先配置训练脚本，可以后面再把数据集补齐。")


def build_data_yaml(data_root: Path, yaml_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    resolved_yaml = yaml_path if yaml_path.is_absolute() else project_root / yaml_path
    resolved_data_root = _resolve_data_root(data_root)

    if resolved_yaml.exists():
        return resolved_yaml

    resolved_yaml.parent.mkdir(parents=True, exist_ok=True)
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    resolved_yaml.write_text(
        f"path: {resolved_data_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"# test: images/test\n\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return resolved_yaml


def _normalize_to_01(images: torch.Tensor) -> torch.Tensor:
    if images.dtype != torch.float32:
        images = images.float()
    if images.max() > 1.5:
        images = images / 255.0
    return images


def apply_photometric_augmentations(images: torch.Tensor) -> torch.Tensor:
    """随机调整亮度、饱和度并注入少量噪点，保持检测框不变。"""

    images = _normalize_to_01(images)
    batch_size = images.shape[0]
    device = images.device

    brightness = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.85, 1.15)
    saturation = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.80, 1.20)
    contrast = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.88, 1.12)
    noise_std = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.00, 0.03)
    gamma = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.90, 1.10)

    images = images * brightness
    mean = images.mean(dim=(2, 3), keepdim=True)
    images = (images - mean) * contrast + mean
    gray = images.mean(dim=1, keepdim=True)
    images = gray + (images - gray) * saturation
    images = images + torch.randn_like(images) * noise_std
    images = torch.pow(images.clamp(1e-6, 1.0), gamma)
    return images.clamp(0.0, 1.0)


def register_training_augmentations(model: YOLO) -> None:
    def _is_primary_process(trainer) -> bool:
        rank = getattr(trainer, "rank", -1)
        return rank in (-1, 0)

    def _safe_float(v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.Tensor):
            if v.numel() == 0:
                return None
            return float(v.detach().mean().item())
        return None

    def _format_loss(trainer) -> str:
        tloss = getattr(trainer, "tloss", None)
        if tloss is None:
            return "loss=N/A"

        if isinstance(tloss, torch.Tensor):
            if tloss.numel() == 0:
                return "loss=N/A"
            values = [float(x) for x in tloss.detach().flatten().tolist()]
        elif isinstance(tloss, (list, tuple)):
            values = []
            for item in tloss:
                fv = _safe_float(item)
                if fv is not None:
                    values.append(fv)
        else:
            fv = _safe_float(tloss)
            values = [fv] if fv is not None else []

        if not values:
            return "loss=N/A"
        if len(values) >= 3:
            return f"loss(box/cls/dfl)={values[0]:.4f}/{values[1]:.4f}/{values[2]:.4f}"
        return "loss=" + "/".join(f"{x:.4f}" for x in values)

    def _format_lr(trainer) -> str:
        lr = getattr(trainer, "lr", None)
        if isinstance(lr, dict) and lr:
            head = []
            for k in sorted(lr.keys())[:3]:
                fv = _safe_float(lr[k])
                if fv is not None:
                    head.append(f"{k}={fv:.6f}")
            if head:
                return "lr(" + ", ".join(head) + ")"

        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            fv = _safe_float(optimizer.param_groups[0].get("lr"))
            if fv is not None:
                return f"lr={fv:.6f}"
        return "lr=N/A"

    def _format_val_metrics(trainer) -> str:
        metrics = getattr(trainer, "metrics", None)
        if not isinstance(metrics, dict) or not metrics:
            return "val=N/A"

        summary: list[str] = []
        preferred = [
            "metrics/mAP50-95(B)",
            "metrics/mAP50(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
        ]

        for key in preferred:
            if key in metrics:
                fv = _safe_float(metrics.get(key))
                if fv is not None:
                    short = key.replace("metrics/", "").replace("(B)", "")
                    summary.append(f"{short}={fv:.4f}")

        if not summary:
            for key in sorted(metrics.keys()):
                fv = _safe_float(metrics.get(key))
                if fv is None:
                    continue
                if key.startswith("metrics/") or key.startswith("val/"):
                    short = key.replace("metrics/", "").replace("val/", "")
                    summary.append(f"{short}={fv:.4f}")
                if len(summary) >= 4:
                    break

        if not summary:
            return "val=N/A"
        return "val(" + ", ".join(summary[:4]) + ")"

    def on_train_start(trainer):
        # 新版 Ultralytics 中 batch 不直接挂在 trainer 上，改为包装 preprocess_batch。
        if not getattr(trainer, "_jomoo_preprocess_patched", False):
            original_preprocess = trainer.preprocess_batch

            def preprocess_with_jomoo_aug(batch):
                batch = original_preprocess(batch)
                if isinstance(batch, dict):
                    images = batch.get("img")
                    if isinstance(images, torch.Tensor):
                        batch["img"] = apply_photometric_augmentations(images)
                return batch

            trainer.preprocess_batch = preprocess_with_jomoo_aug
            trainer._jomoo_preprocess_patched = True

        if not _is_primary_process(trainer):
            return
        total_epochs = getattr(trainer, "epochs", "N/A")
        print("\n[训练开始]", f"epochs={total_epochs}")

    def on_fit_epoch_end(trainer):
        if not _is_primary_process(trainer):
            return
        epoch = _safe_float(getattr(trainer, "epoch", None))
        total_epochs = getattr(trainer, "epochs", "N/A")
        epoch_text = "N/A"
        if epoch is not None:
            epoch_text = str(int(epoch) + 1)
        print(
            "[训练进度]",
            f"epoch={epoch_text}/{total_epochs}",
            _format_loss(trainer),
            _format_lr(trainer),
            _format_val_metrics(trainer),
        )

    def on_train_end(trainer):
        if not _is_primary_process(trainer):
            return
        best = getattr(trainer, "best", None)
        last = getattr(trainer, "last", None)
        best_text = str(best) if best else "N/A"
        last_text = str(last) if last else "N/A"
        print("[训练结束]", f"best={best_text}", f"last={last_text}")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_data_root(data_root: Path) -> Path:
    if data_root.is_absolute():
        return data_root
    return _resolve_project_root() / data_root


def _collect_run_visualizations(run_dir: Path, visualizations_dir: Path) -> list[str]:
    copied_files: list[str] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for file_path in run_dir.glob(pattern):
            target = visualizations_dir / file_path.name
            shutil.copy2(file_path, target)
            copied_files.append(target.name)
    return sorted(set(copied_files))


def _collect_weight_files(run_dir: Path, weights_dir: Path) -> list[str]:
    copied_weights: list[str] = []
    source_weights_dir = run_dir / "weights"
    if not source_weights_dir.exists():
        return copied_weights

    for name in ("best.pt", "last.pt"):
        source = source_weights_dir / name
        if source.exists():
            target = weights_dir / name
            shutil.copy2(source, target)
            copied_weights.append(target.name)
    return copied_weights


def _copy_if_exists(source: Path, target: Path) -> bool:
    if source.exists():
        shutil.copy2(source, target)
        return True
    return False


def organize_artifacts(
    *,
    config: TrainConfig,
    data_yaml: Path,
    run_dir: Path,
    output_root: Path,
    run_tag: str | None,
    train_args: dict[str, Any],
) -> TrainArtifacts:
    tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_root = output_root / config.model_name / tag

    weights_dir = artifact_root / "weights"
    visualizations_dir = artifact_root / "visualizations"
    logs_dir = artifact_root / "logs"
    meta_dir = artifact_root / "meta"
    for folder in (weights_dir, visualizations_dir, logs_dir, meta_dir):
        folder.mkdir(parents=True, exist_ok=True)

    copied_weights = _collect_weight_files(run_dir, weights_dir)
    copied_visuals = _collect_run_visualizations(run_dir, visualizations_dir)
    copied_logs = []
    if _copy_if_exists(run_dir / "results.csv", logs_dir / "results.csv"):
        copied_logs.append("results.csv")
    if _copy_if_exists(run_dir / "args.yaml", logs_dir / "args.yaml"):
        copied_logs.append("args.yaml")

    summary = {
        "model_name": config.model_name,
        "weights": config.weights,
        "run_name": config.run_name,
        "source_run_dir": str(run_dir.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "train_args": train_args,
        "copied_weights": copied_weights,
        "copied_visualizations": copied_visuals,
        "copied_logs": copied_logs,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (meta_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return TrainArtifacts(
        run_dir=run_dir,
        artifact_root=artifact_root,
        weights_dir=weights_dir,
        visualizations_dir=visualizations_dir,
        logs_dir=logs_dir,
        meta_dir=meta_dir,
    )


def train_model(
    config: TrainConfig,
    data_yaml: Path,
    *,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: str | None = DEFAULT_DEVICE,
    workers: int = DEFAULT_WORKERS,
    project: str = DEFAULT_PROJECT,
    seed: int = DEFAULT_SEED,
    patience: int = DEFAULT_PATIENCE,
    optimizer: str = DEFAULT_OPTIMIZER,
    cos_lr: bool = DEFAULT_COS_LR,
    lr0: float = DEFAULT_LR0,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    close_mosaic: int = DEFAULT_CLOSE_MOSAIC,
    mixup: float = DEFAULT_MIXUP,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_tag: str | None = None,
) -> TrainArtifacts:
    set_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.weights)
    register_training_augmentations(model)

    train_args = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": project,
        "seed": seed,
        "patience": patience,
        "optimizer": optimizer,
        "cos_lr": cos_lr,
        "lr0": lr0,
        "weight_decay": weight_decay,
        "close_mosaic": close_mosaic,
        "hsv_h": DEFAULT_HSV_H,
        "hsv_s": DEFAULT_HSV_S,
        "hsv_v": DEFAULT_HSV_V,
        "degrees": DEFAULT_DEGREES,
        "translate": DEFAULT_TRANSLATE,
        "scale": DEFAULT_SCALE,
        "mosaic": DEFAULT_MOSAIC,
        "mixup": mixup,
        "fliplr": DEFAULT_FLIPLR,
    }

    print(f"\n========== 开始训练 {config.model_name} ==========")
    print(f"权重: {config.weights}")
    print(f"数据: {data_yaml}")
    print(f"输出目录: {Path(project) / config.run_name}")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=config.run_name,
        seed=seed,
        cache=False,
        pretrained=True,
        amp=True,
        optimizer=optimizer,
        cos_lr=cos_lr,
        patience=patience,
        lr0=lr0,
        weight_decay=weight_decay,
        hsv_h=DEFAULT_HSV_H,
        hsv_s=DEFAULT_HSV_S,
        hsv_v=DEFAULT_HSV_V,
        degrees=DEFAULT_DEGREES,
        translate=DEFAULT_TRANSLATE,
        scale=DEFAULT_SCALE,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=DEFAULT_FLIPLR,
        mosaic=DEFAULT_MOSAIC,
        mixup=mixup,
        copy_paste=0.0,
        close_mosaic=close_mosaic,
        deterministic=False,
    )

    run_dir = Path(model.trainer.save_dir)
    artifacts = organize_artifacts(
        config=config,
        data_yaml=data_yaml,
        run_dir=run_dir,
        output_root=output_root,
        run_tag=run_tag,
        train_args=train_args,
    )

    print("\n========== 训练产物归档完成 ==========")
    print(f"原始训练目录: {artifacts.run_dir}")
    print(f"归档目录: {artifacts.artifact_root}")
    print(f"模型权重目录: {artifacts.weights_dir}")
    print(f"可视化目录: {artifacts.visualizations_dir}")
    print(f"日志目录: {artifacts.logs_dir}")
    print(f"元信息目录: {artifacts.meta_dir}")
    return artifacts


