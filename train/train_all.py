from __future__ import annotations

import argparse
from datetime import datetime

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
        train_model(
            config,
            data_yaml,
            epochs=defaults.get("epochs", args.epochs),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            seed=args.seed,
            patience=defaults.get("patience", args.patience),
            optimizer=defaults.get("optimizer", args.optimizer),
            cos_lr=defaults.get("cos_lr", args.cos_lr),
            lr0=args.lr0,
            weight_decay=args.weight_decay,
            close_mosaic=defaults.get("close_mosaic", args.close_mosaic),
            mixup=defaults.get("mixup", args.mixup),
            output_root=args.output_root,
            run_tag=f"{shared_tag}_{config.model_name}",
        )


if __name__ == "__main__":
    main()

