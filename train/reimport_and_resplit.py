from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    stem: str
    image: Path
    label: Path


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_resolve_project_root() / path).resolve()


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _load_label_index(labels_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not labels_root.exists():
        return index
    for txt in labels_root.rglob("*.txt"):
        if txt.name.lower() == "classes.txt":
            continue
        index[txt.stem] = txt
    return index


def _collect_existing_pairs(data_root: Path) -> dict[str, Pair]:
    pairs: dict[str, Pair] = {}
    for split in ("train", "val", "test"):
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for image in _iter_images(img_dir):
            label = lbl_dir / f"{image.stem}.txt"
            if label.exists():
                pairs[image.stem] = Pair(stem=image.stem, image=image, label=label)
    return pairs


def _collect_new_pairs(source_images: Path, source_labels: Path) -> tuple[dict[str, Pair], int, int, int]:
    label_index = _load_label_index(source_labels)
    images = _iter_images(source_images)
    paired: dict[str, Pair] = {}
    missing_label = 0

    for image in images:
        label = label_index.get(image.stem)
        if label is None:
            missing_label += 1
            continue
        paired[image.stem] = Pair(stem=image.stem, image=image, label=label)

    return paired, len(images), len(label_index), missing_label


def _backup_current_split(data_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = data_root / f"_split_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for part in ("images", "labels"):
        src = data_root / part
        if src.exists():
            shutil.copytree(src, backup / part, dirs_exist_ok=True)
    return backup


def _reset_split_dirs(data_root: Path) -> None:
    for part in ("images", "labels"):
        base = data_root / part
        base.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            d = base / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def _split_counts(total: int) -> dict[str, int]:
    train = int(total * 0.7)
    val = int(total * 0.2)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def _write_reports(
    report_root: Path,
    *,
    source_images: Path,
    source_labels: Path,
    source_image_count: int,
    source_label_count: int,
    source_paired: int,
    source_missing_label: int,
    merged_total: int,
    backup_dir: Path,
    splits: dict[str, int],
    verify: dict[str, str],
    seed: int,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)

    latest_reimport = report_root / "latest_reimport_report.txt"
    latest_reimport.write_text(
        "\n".join(
            [
                f"SOURCE_IMAGES_ROOT={source_images}",
                f"SOURCE_LABELS_ROOT={source_labels}",
                f"SOURCE_IMAGES={source_image_count}",
                f"SOURCE_LABELS={source_label_count}",
                f"SOURCE_PAIRED={source_paired}",
                f"SOURCE_IMAGES_WITHOUT_LABELS={source_missing_label}",
                f"MERGED_PAIRED_TOTAL={merged_total}",
                f"TRAIN={splits['train']}",
                f"VAL={splits['val']}",
                f"TEST={splits['test']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "source_mode": "merge current project pairs + external paired-only",
        "split_strategy": "A: full reshuffle",
        "report_mode": "C: json + md",
        "seed": seed,
        "source_backup": str(backup_dir),
        "source_images_root": str(source_images),
        "source_labels_root": str(source_labels),
        "source_images": source_image_count,
        "source_labels": source_label_count,
        "source_paired": source_paired,
        "source_images_without_labels": source_missing_label,
        "paired_total": merged_total,
        "splits": splits,
        "verify": verify,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (report_root / "latest_resplit_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# 数据重划分报告",
        "",
        f"- source_mode: {payload['source_mode']}",
        f"- split_strategy: {payload['split_strategy']}",
        f"- report_mode: {payload['report_mode']}",
        f"- source_backup: {payload['source_backup']}",
        f"- seed: {seed}",
        "",
        "## 划分结果 (7:2:1)",
        "",
        "| split | target_count | verify(img/label) |",
        "|---|---:|---:|",
        f"| train | {splits['train']} | {verify['train']} |",
        f"| val | {splits['val']} | {verify['val']} |",
        f"| test | {splits['test']} | {verify['test']} |",
        "",
        f"generated_at: {payload['generated_at']}",
        "",
    ]
    (report_root / "latest_resplit_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将外部图片按 paired-only 并入并按 7:2:1 重划分。")
    parser.add_argument(
        "--source-images",
        type=Path,
        default=Path(r"E:\code\jiumu_product_recognition\data\test_imgs"),
        help="外部图片目录。",
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=Path(r"E:\code\jiumu_product_recognition\data\detect_dataset\labels"),
        help="外部标签根目录（递归查找同名 txt）。",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="项目 data 根目录。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_images = args.source_images.resolve()
    source_labels = args.source_labels.resolve()
    data_root = _resolve(args.data_root)
    report_root = data_root / "_import_reports"

    new_pairs, src_img_count, src_lbl_count, src_img_wo_label = _collect_new_pairs(source_images, source_labels)
    backup_dir = _backup_current_split(data_root)
    existing_pairs = _collect_existing_pairs(backup_dir)
    merged = dict(existing_pairs)
    merged.update(new_pairs)

    all_pairs = list(merged.values())
    random.Random(args.seed).shuffle(all_pairs)

    splits = _split_counts(len(all_pairs))
    train_end = splits["train"]
    val_end = train_end + splits["val"]
    split_map = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }

    _reset_split_dirs(data_root)

    for split, items in split_map.items():
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        for pair in items:
            shutil.copy2(pair.image, img_dir / pair.image.name)
            shutil.copy2(pair.label, lbl_dir / f"{pair.stem}.txt")

    verify: dict[str, str] = {}
    for split in ("train", "val", "test"):
        img_count = len(_iter_images(data_root / "images" / split))
        lbl_count = len([p for p in (data_root / "labels" / split).glob("*.txt") if p.name != "classes.txt"])
        verify[split] = f"{img_count}/{lbl_count}"

    _write_reports(
        report_root,
        source_images=source_images,
        source_labels=source_labels,
        source_image_count=src_img_count,
        source_label_count=src_lbl_count,
        source_paired=len(new_pairs),
        source_missing_label=src_img_wo_label,
        merged_total=len(all_pairs),
        backup_dir=backup_dir,
        splits=splits,
        verify=verify,
        seed=args.seed,
    )

    print("[完成] paired-only 并入 + 7:2:1 重划分")
    print(f"source_images={src_img_count}, source_labels={src_lbl_count}, source_paired={len(new_pairs)}, source_images_without_labels={src_img_wo_label}")
    print(f"merged_total={len(all_pairs)}, splits={splits}, verify={verify}")
    print(f"report={report_root / 'latest_resplit_report.json'}")


if __name__ == "__main__":
    main()

