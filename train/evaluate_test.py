from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from infer.common import find_latest_trained_weight, resolve_path
from ultralytics import YOLO

MODEL_NAMES = ("yolov8s", "yolov9c", "yolov11s")
DEFAULT_DATA_YAML = Path("yolo.yaml")
DEFAULT_TRAINED_ROOT = Path("outputs")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "test_eval"
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 4
DEFAULT_DEVICE = "0"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


def _to_float(value: Any) -> float | Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _resolve_weights(model_name: str, trained_root: Path) -> Path:
    weights = find_latest_trained_weight(trained_root, model_name)
    if weights is None:
        raise FileNotFoundError(
            f"未找到 {model_name} 的训练权重，请先训练后再做 test 评估：{resolve_path(trained_root) / model_name}"
        )
    return weights


def _extract_metrics(result) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            summary[k] = _to_float(v)

    box = getattr(result, "box", None)
    if box is not None:
        summary.setdefault("metrics/mAP50-95(B)", _to_float(getattr(box, "map", None)))
        summary.setdefault("metrics/mAP50(B)", _to_float(getattr(box, "map50", None)))
        summary.setdefault("metrics/precision(B)", _to_float(getattr(box, "mp", None)))
        summary.setdefault("metrics/recall(B)", _to_float(getattr(box, "mr", None)))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 test split 对已训练模型做最终评估。")
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        default=None,
        help="指定单个模型；不传则依次评估三个模型。",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="数据集配置文件，默认 yolo.yaml。",
    )
    parser.add_argument(
        "--trained-root",
        type=Path,
        default=DEFAULT_TRAINED_ROOT,
        help="训练产物根目录，默认 outputs。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="test 评估结果输出根目录，默认 outputs/test_eval。",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="评估图片尺寸。")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="评估 batch size。")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="评估设备，例如 0 或 cpu。")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值。")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU 阈值。")
    parser.add_argument("--run-tag", type=str, default=None, help="评估目录标识，不传则用时间戳。")
    return parser.parse_args()


def run_test_eval_for_model(
    *,
    model_name: str,
    data_yaml: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_tag: str | None = None,
    weights: Path | None = None,
    trained_root: Path = DEFAULT_TRAINED_ROOT,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: str = DEFAULT_DEVICE,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
) -> Path:
    resolved_data_yaml = resolve_path(data_yaml)
    resolved_output_root = resolve_path(output_root)
    resolved_weights = resolve_path(weights) if weights is not None else _resolve_weights(model_name, trained_root)
    eval_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    model = YOLO(str(resolved_weights))
    eval_dir = resolved_output_root / model_name / eval_tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== test 评估 {model_name} ==========")
    print(f"权重: {resolved_weights}")
    print(f"数据: {resolved_data_yaml}")
    print(f"输出目录: {eval_dir}")

    result = model.val(
        data=str(resolved_data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        project=str(eval_dir),
        name="raw",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    summary = {
        "model_name": model_name,
        "weights": str(resolved_weights),
        "data_yaml": str(resolved_data_yaml),
        "split": "test",
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "conf": conf,
        "iou": iou,
        "metrics": _extract_metrics(result),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    meta_dir = eval_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[test 评估完成] summary={summary_path}")
    return summary_path


def _evaluate_one(
    *,
    model_name: str,
    data_yaml: Path,
    trained_root: Path,
    output_root: Path,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    run_tag: str,
) -> Path:
    return run_test_eval_for_model(
        model_name=model_name,
        data_yaml=data_yaml,
        output_root=output_root,
        run_tag=run_tag,
        trained_root=trained_root,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
    )


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_names = (args.model_name,) if args.model_name else MODEL_NAMES

    for model_name in model_names:
        _evaluate_one(
            model_name=model_name,
            data_yaml=args.data_yaml,
            trained_root=args.trained_root,
            output_root=args.output_root,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            run_tag=f"{run_tag}_{model_name}",
        )


if __name__ == "__main__":
    main()

