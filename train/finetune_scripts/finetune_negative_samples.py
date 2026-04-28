# -*- coding: utf-8 -*-
"""
YOLOv9c 负样本抑制微调脚本

创建时间: 2026-04-24 10:37
基础权重: outputs/yolov9c/v9c_cos_lr_mixup_opt/weights/best.pt

微调目标:
  - 通过引入负样本降低背景误检率
  - 提升模型对背景的鲁棒性
  - 减少将货架、包装箱误认为产品的情况

关键配置:
  - 学习率: 0.0001 (较低，避免破坏已有知识)
  - 冻结层数: 10 (保留底层特征提取能力)
  - 批次大小: 8
  - 早停耐心值: 5
  - 余弦学习率退火: True
  - 置信度阈值: 0.65 (训练时使用较高阈值)

负样本特点:
  - 空标签文件（不含任何目标）
  - 包含货架背景、包装箱等非产品场景
  - 帮助模型学习“什么不是产品”

预期效果:
  - 显著降低背景误检
  - 提升模型判别能力
  - 可能略微降低召回率

使用方法:
  python train/finetune_scripts/finetune_negative_samples.py

输出位置:
  outputs/yolov9c/v9c_finetune_negative_samples/weights/best.pt
"""

from ultralytics import YOLO

def finetune_with_negative_samples():
    """使用负样本进行微调"""
    
    # 加载当前最佳模型
    model_path = 'outputs/yolov9c/v9c_cos_lr_mixup_opt/weights/best.pt'
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 训练配置
    print("\n开始微调训练...")
    print("=" * 80)
    print("训练配置:")
    print(f"  数据配置文件: data.yaml")
    print(f"  训练轮数: 20 epochs")
    print(f"  图像尺寸: 640")
    print(f"  批次大小: 8")
    print(f"  初始学习率: 0.0001")
    print(f"  冻结层数: 10 (冻结前10层)")
    print(f"  早停耐心值: 5")
    print(f"  置信度阈值: 0.65")
    print(f"  输出目录: outputs/yolov9c/v9c_finetune_negative_samples")
    print("=" * 80)
    
    # 执行训练
    results = model.train(
        data='data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        lr0=0.0001,
        freeze=10,
        patience=5,
        conf=0.65,
        project='outputs/yolov9c',
        name='v9c_finetune_negative_samples',
        device=0,
        verbose=True,
        workers=4,
        optimizer='AdamW',
        weight_decay=0.0005,
        momentum=0.937,
        cos_lr=True,  # 使用余弦学习率退火
        close_mosaic=5,  # 最后5个epoch关闭mosaic增强
    )
    
    print("\n" + "=" * 80)
    print("✅ 微调训练完成！")
    print("=" * 80)
    print(f"\n模型保存位置:")
    print(f"  best.pt: outputs/yolov9c/v9c_finetune_negative_samples/weights/best.pt")
    print(f"  last.pt: outputs/yolov9c/v9c_finetune_negative_samples/weights/last.pt")
    print(f"\n训练结果:")
    print(f"  最佳mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  最佳mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    try:
        results = finetune_with_negative_samples()
        print("\n🎉 训练成功完成！")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
