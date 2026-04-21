from __future__ import annotations

import argparse
from datetime import datetime

from infer.common import (
    InferConfig,
    add_common_cli_args,
    import_inference_images,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顺序执行 YOLOv8s、YOLOv9c、YOLOv11s 三模型推理。")
    add_common_cli_args(parser)
    parser.add_argument("--weights-v8", type=str, default=None, help="YOLOv8s 权重路径；不传则自动找最新训练产物。")
    parser.add_argument("--weights-v9", type=str, default=None, help="YOLOv9c 权重路径；不传则自动找最新训练产物。")
    parser.add_argument("--weights-v11", type=str, default=None, help="YOLOv11s 权重路径；不传则自动找最新训练产物。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = args.source.resolve()
    if not args.skip_import:
        source_dir = import_inference_images(args.source, args.import_root)

    shared_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    configs = (
        InferConfig(model_name="yolov8s", weights=args.weights_v8),
        InferConfig(model_name="yolov9c", weights=args.weights_v9),
        InferConfig(model_name="yolov11s", weights=args.weights_v11),
    )

    for config in configs:
        run_inference(
            config,
            source_dir=source_dir,
            data_yaml=args.data_yaml,
            trained_root=args.trained_root,
            output_root=args.output_root,
            run_tag=f"{shared_tag}_{config.model_name}",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            line_width=args.line_width,
            font_scale=args.font_scale,
            font_thickness=args.font_thickness,
        )


if __name__ == "__main__":
    main()

