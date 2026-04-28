# -*- coding: utf-8 -*-
"""
YOLOv9c 角阀类专项微调脚本

创建时间: 2026-04-24 14:34
基础权重: outputs/yolov9c/v9c_finetune_negative_samples/weights/best.pt (优先)
           outputs/yolov9c/v9c_cos_lr_mixup_opt/weights/best.pt (备选)
           yolov9c.pt (最后备选)

微调目标:
  - 专门针对九牧安全角阀进行优化
  - 提高安全角阀的类别权重至 2.0
  - 小轮次精细微调（15 epochs）

关键配置:
  - 学习率: 0.00005 (极低，精细调整)
  - 冻结层数: 15 (保留底层特征)
  - 批次大小: 8
  - 早停耐心值: 3 (快速收敛)
  - 余弦学习率退火: True
  - Box损失权重: 7.5 (提高定位精度)

预期效果:
  - 提升安全角阀检测准确率
  - 减少与其他角阀类别的混淆
  - 改善密集排列场景的检测

使用方法:
  python train/finetune_scripts/finetune_angle_valve_specialist.py

输出位置:
  outputs/yolov9c/v9c_angle_valve_specialist/weights/best.pt
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def finetune_angle_valve():
    """针对安全角阀进行微调"""
    
    # 尝试加载可用的最佳模型
    model_candidates = [
        'outputs/yolov9c/v9c_finetune_negative_samples/weights/best.pt',
        'outputs/yolov9c/v9c_cos_lr_mixup_opt/weights/best.pt',
        'yolov9c.pt'
    ]
    
    model_path = None
    for candidate in model_candidates:
        if Path(candidate).exists():
            model_path = candidate
            print(f"找到模型: {model_path}")
            break
    
    if model_path is None:
        print("错误: 找不到任何可用的模型文件")
        print("请确保以下路径之一存在:")
        for candidate in model_candidates:
            print(f"  - {candidate}")
        return None
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 读取data.yaml获取类别信息
    with open('data.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    class_names_dict = data_config['names']
    
    # 处理names可能是字典或列表的情况
    if isinstance(class_names_dict, dict):
        num_classes = len(class_names_dict)
        class_names_list = list(class_names_dict.values())
    else:
        num_classes = len(class_names_dict)
        class_names_list = class_names_dict
    
    print(f"类别数量: {num_classes}")
    print(f"类别列表: {class_names_dict}")
    
    # 找到"九牧安全角阀"的索引
    angle_valve_idx = None
    for idx, name in enumerate(class_names_list):
        if isinstance(name, str) and '安全角阀' in name:
            angle_valve_idx = idx
            print(f"\n找到目标类别: '{name}' (索引: {idx})")
            break
    
    if angle_valve_idx is None:
        print("警告: 未找到'安全角阀'类别，使用默认权重")
        class_weights = None
    else:
        # 设置类别权重：安全角阀设为2.0，其他为1.0
        class_weights = [1.0] * num_classes
        class_weights[angle_valve_idx] = 2.0
        print(f"类别权重已设置: 安全角阀=2.0, 其他=1.0")
    
    # 训练配置
    print("\n开始专项微调训练...")
    print("=" * 80)
    print("训练配置:")
    print(f"  数据配置文件: data.yaml")
    print(f"  训练轮数: 15 epochs (小轮次)")
    print(f"  图像尺寸: 640")
    print(f"  批次大小: 8")
    print(f"  初始学习率: 0.00005 (更低的学习率)")
    print(f"  冻结层数: 15 (冻结更多层)")
    print(f"  早停耐心值: 3")
    print(f"  类别权重: 安全角阀=2.0" if class_weights else "  类别权重: 默认")
    print(f"  输出目录: outputs/yolov9c/v9c_angle_valve_specialist")
    print("=" * 80)
    
    # 执行训练
    results = model.train(
        data='data.yaml',
        epochs=15,
        imgsz=640,
        batch=8,
        lr0=0.00005,  # 更低的学习率
        freeze=15,    # 冻结更多层
        patience=3,   # 更短的早停耐心值
        project='outputs/yolov9c',
        name='v9c_angle_valve_specialist',
        device=0,
        verbose=True,
        workers=4,
        optimizer='AdamW',
        weight_decay=0.0005,
        momentum=0.937,
        cos_lr=True,  # 余弦学习率退火
        close_mosaic=3,  # 最后3个epoch关闭mosaic
        box=7.5,  # 提高box损失权重
        cls=0.5,  # 降低cls损失权重
    )
    
    print("\n" + "=" * 80)
    print("✅ 专项微调训练完成！")
    print("=" * 80)
    print(f"\n模型保存位置:")
    print(f"  best.pt: outputs/yolov9c/v9c_angle_valve_specialist/weights/best.pt")
    print(f"  last.pt: outputs/yolov9c/v9c_angle_valve_specialist/weights/last.pt")
    print(f"\n训练结果:")
    print(f"  最佳mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  最佳mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  最佳Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  最佳Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    try:
        results = finetune_angle_valve()
        print("\n🎉 专项微调成功完成！")
        print("\n💡 下一步:")
        print("  1. 使用新模型进行推理测试")
        print("  2. 检查安全角阀的检测效果")
        print("  3. 如有需要，调整阈值配置")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
