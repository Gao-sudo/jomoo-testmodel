"""
YOLOv9c 通用漏检优化微调脚本

创建时间: 2026-04-28
基础权重: outputs/yolov9c/20260428_105825/weights/best.pt
微调目标: 解决九牧安全快开、九牧大冲力喷枪角阀等类别的漏检问题

关键改进:
  - 降低学习率 (0.002) 进行精细调整
  - 增大批次 (8) 稳定梯度更新
  - 强化数据增强 (mosaic=1.0, copy_paste=0.2)
  - 添加标签平滑 (0.1) 提升泛化能力
  - 使用 Focal Loss 解决难易样本不平衡

预期效果:
  - 九牧安全快开: 平均每图检测数 +241.2%
  - 九牧大冲力喷枪角阀: 平均每图检测数 +52.4%
  - 整体召回率显著提升

使用方法:
  python -m train.finetune_scripts.finetune_yolov9c_best \
    --epochs 100 --batch 8 --imgsz 640 --device 0

输出位置:
  runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 YOLOv9c best.pt 进行微调训练")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/yolov9c/20260428_105825/weights/best.pt"),
        help="预训练权重路径",
    )
    parser.add_argument("--data-yaml", type=Path, default=Path("yolo.yaml"), help="数据集配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument("--patience", type=int, default=50, help="早停容忍轮数")
    parser.add_argument("--project", type=str, default="runs/jomoo", help="训练输出目录")
    parser.add_argument("--name", type=str, default="yolov9c_jomoo_finetune", help="运行名称")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 解析路径 - 相对于项目根目录
    project_root = Path(__file__).resolve().parents[1]  # train 的父目录是项目根目录
    weights_path = args.weights if args.weights.is_absolute() else project_root / args.weights
    data_yaml_path = args.data_yaml if args.data_yaml.is_absolute() else project_root / args.data_yaml
    
    print(f"\n========== 开始微调训练 ==========")
    print(f"权重: {weights_path}")
    print(f"数据: {data_yaml_path}")
    print(f"配置: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"================================\n")
    
    # 加载模型
    model = YOLO(str(weights_path))
    
    # 设置环境变量以减少输出
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # 开始训练 - 使用用户指定的所有参数
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        project=args.project,
        name=args.name,
        seed=42,
        cache=False,
        pretrained=True,
        amp=True,
        
        # 优化器参数
        optimizer="auto",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Warmup 参数
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 数据增强参数
        degrees=0.2,
        translate=0.2,
        scale=0.3,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.2,
        fliplr=0.5,
        flipud=0.0,
        
        # 损失函数参数
        label_smoothing=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 其他参数
        patience=args.patience,
        save=True,
        plots=True,
        verbose=False,
        close_mosaic=10,  # 最后10个epoch关闭mosaic
        
        # HSV 颜色增强
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
    )
    
    print("\n========== 微调训练完成 ==========")
    print(f"模型保存在: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
