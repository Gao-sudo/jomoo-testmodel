from __future__ import annotations

import argparse

from infer.common import (
    InferConfig,
    add_common_cli_args,
    apply_single_model_cli_args,
    import_inference_images,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLOv11s 对导入图片进行批量推理。")
    add_common_cli_args(parser)
    apply_single_model_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    if not args.skip_import:
        source_dir = import_inference_images(args.source, args.import_root)

    config = InferConfig(model_name="yolov11s", weights=args.weights)
    run_inference(
        config,
        source_dir=source_dir,
        data_yaml=args.data_yaml,
        trained_root=args.trained_root,
        output_root=args.output_root,
        run_tag=args.run_tag,
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

