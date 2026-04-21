from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml
from ultralytics import YOLO

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE = Path(r"E:\code\jiumu_product_recognition\data\test_imgs")
DEFAULT_IMPORT_ROOT = Path("data") / "infer_images"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "infer"
DEFAULT_TRAINED_ROOT = Path("outputs")
DEFAULT_DATA_YAML = Path("data.yaml")
DEFAULT_DEVICE = "0"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_LINE_WIDTH = 1
DEFAULT_FONT_SCALE = 0.45
DEFAULT_FONT_THICKNESS = 1

# Avoid blue/white tones because the shelf background is blue-white.
BOX_COLOR_PALETTE_BGR: list[tuple[int, int, int]] = [
    (32, 32, 220),   # red
    (32, 170, 32),   # green
    (40, 120, 235),  # orange
    (180, 60, 180),  # purple
    (20, 190, 190),  # yellow-ish
    (90, 40, 200),   # magenta-ish
]


@dataclass(frozen=True)
class InferConfig:
    model_name: str
    weights: str | None = None


@dataclass(frozen=True)
class InferArtifacts:
    input_dir: Path
    run_dir: Path
    visualizations_dir: Path
    labels_dir: Path
    json_path: Path
    summary_path: Path


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="原始推理图片目录，默认 E:\\code\\jiumu_product_recognition\\data\\test_imgs。",
    )
    parser.add_argument(
        "--import-root",
        type=Path,
        default=DEFAULT_IMPORT_ROOT,
        help="导入到项目内的目标根目录。",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="跳过导入步骤，直接使用 --source 目录推理。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="推理结果根目录，默认 outputs/infer。",
    )
    parser.add_argument(
        "--trained-root",
        type=Path,
        default=DEFAULT_TRAINED_ROOT,
        help="训练产物根目录，默认 outputs；自动权重会从这里查找最新 best.pt。",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="类别配置文件，默认 data.yaml。",
    )
    parser.add_argument("--run-tag", type=str, default=None, help="本次推理标签，不传则用时间戳。")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="推理设备，例如 0 或 cpu。")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理尺寸。")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值。")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU 阈值。")
    parser.add_argument("--max-det", type=int, default=300, help="每张图最大检测数量。")
    parser.add_argument("--line-width", type=int, default=DEFAULT_LINE_WIDTH, help="可视化框线宽。")
    parser.add_argument("--font-scale", type=float, default=DEFAULT_FONT_SCALE, help="可视化文字大小。")
    parser.add_argument("--font-thickness", type=int, default=DEFAULT_FONT_THICKNESS, help="可视化文字线宽。")


def apply_single_model_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_weights: str | None = None,
) -> None:
    parser.add_argument("--weights", type=str, default=default_weights, help="模型权重路径。")


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_resolve_project_root() / path).resolve()


def _default_pretrained_weights(model_name: str) -> Path:
    mapping = {
        "yolov8s": Path("yolov8s.pt"),
        "yolov9c": Path("yolov9c.pt"),
        "yolov11s": Path("yolo11s.pt"),
    }
    try:
        return mapping[model_name]
    except KeyError as exc:
        raise ValueError(f"不支持的模型名: {model_name}") from exc


def _candidate_run_score(run_dir: Path) -> float:
    summary = run_dir / "meta" / "summary.json"
    best_pt = run_dir / "weights" / "best.pt"
    if summary.exists():
        return summary.stat().st_mtime
    if best_pt.exists():
        return best_pt.stat().st_mtime
    return run_dir.stat().st_mtime


def find_latest_trained_weight(trained_root: Path, model_name: str) -> Path | None:
    resolved_root = resolve_path(trained_root)
    model_root = resolved_root / model_name
    if not model_root.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for run_dir in model_root.iterdir():
        if not run_dir.is_dir():
            continue
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            continue
        candidates.append((_candidate_run_score(run_dir), best_pt))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resolve_inference_weights(
    *,
    model_name: str,
    explicit_weights: str | None,
    trained_root: Path,
) -> tuple[Path, str]:
    if explicit_weights:
        weight_path = resolve_path(Path(explicit_weights))
        if not weight_path.exists():
            raise FileNotFoundError(f"权重不存在: {weight_path}")
        return weight_path, "explicit"

    latest = find_latest_trained_weight(trained_root, model_name)
    if latest is not None:
        return latest.resolve(), "latest_trained"

    fallback = resolve_path(_default_pretrained_weights(model_name))
    if not fallback.exists():
        raise FileNotFoundError(f"找不到可用权重：{fallback}")
    return fallback, "pretrained_fallback"


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(files)


def import_inference_images(source: Path, import_root: Path) -> Path:
    src = source.resolve()
    if not src.exists():
        raise FileNotFoundError(f"推理源目录不存在: {src}")

    target_root = resolve_path(import_root)
    target_dir = target_root / src.name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in list_images(src):
        target_path = target_dir / image_path.name
        stem = image_path.stem
        suffix = image_path.suffix
        idx = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{idx}{suffix}"
            idx += 1
        shutil.copy2(image_path, target_path)
        copied += 1

    print(f"[导入完成] source={src} -> target={target_dir}, images={copied}")
    return target_dir


def _to_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _resolve_data_yaml(data_yaml: Path) -> Path:
    return resolve_path(data_yaml)


def _load_class_names_from_data_yaml(data_yaml: Path) -> list[str]:
    yaml_path = _resolve_data_yaml(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml 不存在: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        ordered: list[tuple[int, str]] = []
        for k, v in names.items():
            ordered.append((int(k), str(v)))
        ordered.sort(key=lambda item: item[0])
        return [v for _, v in ordered]
    raise ValueError(f"data.yaml 的 names 格式无效: {yaml_path}")


def ensure_classes_txt_files(data_yaml: Path, labels_root: Path = Path("data") / "labels") -> list[Path]:
    class_names = _load_class_names_from_data_yaml(data_yaml)
    resolved_labels_root = resolve_path(labels_root)
    if not resolved_labels_root.exists():
        print(f"[提示] labels 根目录不存在，跳过 classes.txt 生成: {resolved_labels_root}")
        return []

    content = "\n".join(class_names) + "\n"
    written_files: list[Path] = []
    for sub in sorted(resolved_labels_root.iterdir()):
        if not sub.is_dir():
            continue
        classes_file = sub / "classes.txt"
        classes_file.write_text(content, encoding="utf-8")
        written_files.append(classes_file)

    if written_files:
        print("[已更新 classes.txt]", ", ".join(str(p) for p in written_files))
    else:
        print(f"[提示] labels 下没有子目录，未生成 classes.txt: {resolved_labels_root}")
    return written_files


def _color_for_class(cls_idx: int) -> tuple[int, int, int]:
    return BOX_COLOR_PALETTE_BGR[cls_idx % len(BOX_COLOR_PALETTE_BGR)]


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color
    return r, g, b


@lru_cache(maxsize=1)
def _load_chinese_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        fonts_dir / "msyh.ttc",
        fonts_dir / "msyhbd.ttc",
        fonts_dir / "simhei.ttf",
        fonts_dir / "simsun.ttc",
        fonts_dir / "NotoSansCJK-Regular.ttc",
        fonts_dir / "NotoSansCJKsc-Regular.otf",
    ]
    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except OSError:
                continue

    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _safe_write_image(path: Path, image) -> None:
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"保存可视化图片失败: {path}")
    encoded.tofile(str(path))


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text, font=font, stroke_width=1)
    width, height = draw.textsize(text, font=font)
    return 0, 0, width, height


def _render_custom_visualization(
    result,
    *,
    visualizations_dir: Path,
    line_width: int,
    font_scale: float,
    font_thickness: int,
) -> Path:
    image = result.orig_img.copy()
    boxes = result.boxes
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_size = max(12, int(round(18 * font_scale)))
    font = _load_chinese_font(font_size)

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
        clss = boxes.cls.cpu().tolist() if boxes.cls is not None else [0.0] * len(xyxy)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cls_idx = int(clss[i])
            conf = _to_float(confs[i])
            name = result.names.get(cls_idx, str(cls_idx)) if isinstance(result.names, dict) else str(cls_idx)
            color = _color_for_class(cls_idx)
            rgb = _bgr_to_rgb(color)

            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=max(1, line_width))

            label = f"{name} {conf:.2f}"
            bbox = _text_bbox(draw, label, font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            base = max(2, int(round(font_size * 0.18)))
            text_x = max(0, x1)
            text_y = max(text_h + 2, y1 - 4)
            bg_top = max(0, text_y - text_h - base - 2)
            bg_bottom = min(pil_image.height - 1, text_y + 2)
            bg_right = min(pil_image.width - 1, text_x + text_w + 4)

            draw.rectangle([text_x, bg_top, bg_right, bg_bottom], fill=rgb)
            draw.text(
                (text_x + 2, bg_top + 1),
                label,
                font=font,
                fill=(255, 255, 255),
                stroke_width=max(1, font_thickness),
                stroke_fill=(0, 0, 0),
            )

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    output_path = visualizations_dir / Path(str(result.path)).name
    _safe_write_image(output_path, image)
    return output_path


def _serialize_result(result) -> dict[str, Any]:
    boxes = result.boxes
    items: list[dict[str, Any]] = []

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().tolist()
        xywhn = boxes.xywhn.cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
        clss = boxes.cls.cpu().tolist() if boxes.cls is not None else [0.0] * len(xyxy)

        for i in range(len(xyxy)):
            cls_idx = int(clss[i])
            name = result.names.get(cls_idx, str(cls_idx)) if isinstance(result.names, dict) else str(cls_idx)
            items.append(
                {
                    "cls": cls_idx,
                    "name": name,
                    "conf": round(_to_float(confs[i]), 6),
                    "xyxy": [round(float(v), 3) for v in xyxy[i]],
                    "xywhn": [round(float(v), 6) for v in xywhn[i]],
                }
            )

    return {
        "image": str(result.path),
        "detections": items,
    }


def run_inference(
    config: InferConfig,
    *,
    source_dir: Path,
    data_yaml: Path,
    trained_root: Path,
    output_root: Path,
    run_tag: str | None,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    max_det: int,
    line_width: int,
    font_scale: float,
    font_thickness: int,
) -> InferArtifacts:
    source_dir = source_dir.resolve()
    output_root = resolve_path(output_root)
    tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = output_root / config.model_name / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(source_dir)
    if not images:
        raise RuntimeError(f"目录中没有可推理图片: {source_dir}")

    weights_path, weights_source = resolve_inference_weights(
        model_name=config.model_name,
        explicit_weights=config.weights,
        trained_root=trained_root,
    )

    visualizations_dir = run_dir / "visualizations"
    labels_dir = run_dir / "predictions_txt"
    meta_dir = run_dir / "meta"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== 开始推理 {config.model_name} ==========")
    print(f"权重: {weights_path}")
    print(f"权重来源: {weights_source}")
    print(f"输入目录: {source_dir}")
    print(f"输出目录: {run_dir}")
    print(f"图片数量: {len(images)}")

    written_classes = ensure_classes_txt_files(data_yaml)

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(run_dir),
        name="raw",
        exist_ok=True,
        verbose=True,
    )

    for item in results:
        _render_custom_visualization(
            item,
            visualizations_dir=visualizations_dir,
            line_width=line_width,
            font_scale=font_scale,
            font_thickness=font_thickness,
        )

    raw_labels_dir = run_dir / "raw" / "labels"
    if raw_labels_dir.exists():
        for txt_file in raw_labels_dir.glob("*.txt"):
            shutil.copy2(txt_file, labels_dir / txt_file.name)

    predictions = [_serialize_result(item) for item in results]
    total_dets = sum(len(item["detections"]) for item in predictions)

    class_counter: dict[str, int] = {}
    for item in predictions:
        for det in item["detections"]:
            name = det["name"]
            class_counter[name] = class_counter.get(name, 0) + 1

    json_path = run_dir / "predictions.json"
    json_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "model_name": config.model_name,
        "weights": str(weights_path),
        "weights_source": weights_source,
        "source_dir": str(source_dir),
        "run_dir": str(run_dir),
        "image_count": len(images),
        "det_count": total_dets,
        "class_counter": class_counter,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": device,
        "max_det": max_det,
        "line_width": line_width,
        "font_scale": font_scale,
        "font_thickness": font_thickness,
        "classes_txt_files": [str(p) for p in written_classes],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = meta_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[推理完成]", f"detections={total_dets}", f"summary={summary_path}")
    return InferArtifacts(
        input_dir=source_dir,
        run_dir=run_dir,
        visualizations_dir=visualizations_dir,
        labels_dir=labels_dir,
        json_path=json_path,
        summary_path=summary_path,
    )

