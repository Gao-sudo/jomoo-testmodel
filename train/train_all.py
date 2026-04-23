from __future__ import annotations

import argparse
from datetime import datetime
from typing import cast

from train.common import (
    TrainConfig,
    add_common_cli_args,
    build_data_yaml,
    resolve_model_defaults,
    train_model,
    validate_data_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="顺序训练 YOLOv8s、YOLOv9c、YOLOv11s，并统一归档训练产物。"
    )
    add_common_cli_args(parser)
    parser.set_defaults(patience=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_data_layout(args.data_root)
    data_yaml = build_data_yaml(args.data_root, args.data_yaml)

    shared_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    configs = (
        TrainConfig(model_name="yolov8s", weights="yolov8s.pt", run_name="yolov8s_jomoo"),
        TrainConfig(model_name="yolov9c", weights="yolov9c.pt", run_name="yolov9c_jomoo"),
        TrainConfig(model_name="yolov11s", weights="yolo11s.pt", run_name="yolov11s_jomoo"),
    )

    for config in configs:
        defaults = resolve_model_defaults(config.model_name)
        patience = args.patience if args.patience is not None else defaults.get("patience", 30)
        epochs: int = args.epochs
        if "epochs" in defaults:
            epochs = int(cast(int, defaults["epochs"]))

        optimizer: str = args.optimizer
        if "optimizer" in defaults:
            optimizer = str(cast(str, defaults["optimizer"]))

        cos_lr: bool = args.cos_lr
        if "cos_lr" in defaults:
            cos_lr = bool(cast(bool, defaults["cos_lr"]))

        close_mosaic: int = args.close_mosaic
        if "close_mosaic" in defaults:
            close_mosaic = int(cast(int, defaults["close_mosaic"]))

        mixup: float = args.mixup
        if "mixup" in defaults:
            mixup = float(cast(float, defaults["mixup"]))

        lr0: float = args.lr0
        if "lr0" in defaults:
            lr0 = float(cast(float, defaults["lr0"]))

        weight_decay: float = args.weight_decay
        if "weight_decay" in defaults:
            weight_decay = float(cast(float, defaults["weight_decay"]))

        train_model(
            config,
            data_yaml,
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            seed=args.seed,
            patience=patience,
            optimizer=optimizer,
            cos_lr=cos_lr,
            lr0=lr0,
            weight_decay=weight_decay,
            close_mosaic=close_mosaic,
            mixup=mixup,
            output_root=args.output_root,
            run_tag=f"{shared_tag}_{config.model_name}",
        )


if __name__ == "__main__":
    main()

