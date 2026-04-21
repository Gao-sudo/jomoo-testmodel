from __future__ import annotations

import argparse

from train.common import (
    apply_model_defaults,
    TrainConfig,
    add_common_cli_args,
    build_data_yaml,
    train_model,
    validate_data_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练九牧货架商品检测 YOLOv8s 模型。")
    add_common_cli_args(parser)
    apply_model_defaults(parser, "yolov8s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_data_layout(args.data_root)
    data_yaml = build_data_yaml(args.data_root, args.data_yaml)
    config = TrainConfig(
        model_name="yolov8s",
        weights="yolov8s.pt",
        run_name="yolov8s_jomoo",
    )
    train_model(
        config,
        data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        seed=args.seed,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        output_root=args.output_root,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()

