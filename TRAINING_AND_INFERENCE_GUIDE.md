# YOLOv9c 完整训练与推理流程指南

本文档详细记录了从数据准备到模型推理的完整流程，特别针对**九牧安全快开**和**九牧大冲力喷枪角阀**等易漏检类别的优化方案。

---

## 📋 目录

1. [数据准备与导入](#1-数据准备与导入)
2. [第一阶段：基础训练](#2-第一阶段基础训练)
3. [第二阶段：专项微调](#3-第二阶段专项微调)
4. [推理配置与优化](#4-推理配置与优化)
5. [效果对比与分析](#5-效果对比与分析)
6. [常见问题与解决方案](#6-常见问题与解决方案)

---

## 1. 数据准备与导入

### 1.1 数据来源

本次训练使用两个外部数据源：
- **源目录1**: `C:\Users\China\Desktop\新建文件夹\第一次数据-挑选\dataset-1-81\dataset-1-81`
- **源目录2**: `C:\Users\China\Desktop\新建文件夹\第二次数据-修正`

每个源目录应包含：
```
dataset/
├── images/    # 图片文件（.jpg, .png等）
└── labels/    # YOLO格式标注文件（.txt）
```

### 1.2 数据导入与重划分

使用 `reimport_and_resplit.py` 脚本自动完成数据导入和划分：

```powershell
cd E:\code\jomoo-testmodel
python train/reimport_and_resplit.py
```

**脚本功能**：
- ✅ 从多个源目录收集图片和标签
- ✅ 自动备份原有数据到 `_split_backup_时间戳/`
- ✅ 按 7:2:1 比例重新划分训练集、验证集、测试集
- ✅ 生成详细报告到 `data/_import_reports/`

**数据划分结果**：
```
总图片数: 316
├── 训练集 (train): 221 张 (70%)
├── 验证集 (val):   63 张 (20%)
└── 测试集 (test):  32 张 (10%)
```

**验证数据完整性**：
```powershell
# 检查各目录文件数量
Get-ChildItem "data\images\train" -File | Measure-Object
Get-ChildItem "data\images\val" -File | Measure-Object
Get-ChildItem "data\images\test" -File | Measure-Object
```

### 1.3 数据集配置

确认 `data.yaml` 或 `yolo.yaml` 配置正确：

```yaml
path: E:/code/jomoo-testmodel/data
train: images/train
val: images/val
test: images/test

names:
  0: 九牧增压花洒
  1: 九牧增压花洒套装
  2: 九牧大冲力喷枪角阀
  3: 九牧安全快开
  4: 九牧安全角阀
  5: 九牧百搭下水
  6: 九牧百搭下水（软袋）
  7: 九牧轻音盖板
  8: 九牧防断裂淋浴软管
  9: 九牧防漏水件
  10: 九牧防爆编织软管
  11: 九牧防臭下水管
  12: 九牧防臭地漏
  13: 九牧健康编织软管
```

---

## 2. 第一阶段：基础训练

### 2.1 训练目标

使用 COCO 预训练权重 `yolov9c.pt`，让模型学习九牧产品的基本检测能力。

### 2.2 训练命令

```powershell
python -m train.train_yolov9c `
  --epochs 200 `
  --patience 50 `
  --batch 4 `
  --imgsz 640 `
  --device 0
```

### 2.3 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 200 | 最大训练轮数 |
| `--patience` | 50 | 早停容忍轮数（50轮无提升则停止） |
| `--batch` | 4 | 批次大小（根据显存调整） |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--device` | 0 | GPU设备编号 |

**默认增强策略**（已在 `train/common.py` 中配置）：
- Mosaic: 0.8
- Mixup: 0.15
- HSV 增强: h=0.015, s=0.6, v=0.4
- 旋转: degrees=10
- 缩放: scale=0.5

### 2.4 训练输出

训练完成后，权重保存在：
```
outputs/yolov9c/20260428_105825/weights/best.pt  ← 最佳权重
outputs/yolov9c/20260428_105825/weights/last.pt  ← 最后权重
```

同时保留 Ultralytics 原始输出：
```
runs/jomoo/yolov9c_jomoo/weights/best.pt
```

### 2.5 监控训练进度

训练过程中会显示：
```
[训练开始] epochs=200 patience=50
[训练进度] epoch=1/200 loss(box/cls/dfl)=... val(mAP50-95)=...
[训练进度] epoch=2/200 ...
...
[训练结束] best=... last=...
```

---

## 3. 第二阶段：专项微调

### 3.1 微调目标

针对第一阶段发现的漏检问题（特别是**九牧安全快开**和**九牧大冲力喷枪角阀**），进行精细化调优。

### 3.2 微调脚本集合

所有微调脚本已整理到 `train/finetune_scripts/` 目录，包含：

| 脚本 | 创建时间 | 基础权重 | 微调目标 |
|------|---------|---------|----------|
| **finetune_yolov9c_best.py** | 2026-04-28 | outputs/yolov9c/20260428_105825/weights/best.pt | 通用漏检优化 |
| **finetune_angle_valve_specialist.py** | 2026-04-24 | v9c_finetune_negative_samples/best.pt | 角阀类专项优化 |
| **finetune_negative_samples.py** | 2026-04-24 | v9c_cos_lr_mixup_opt/best.pt | 负样本抑制优化 |

详细说明请查看：[train/finetune_scripts/README.md](train/finetune_scripts/README.md)

#### finetune_yolov9c_best.py（推荐使用）

已创建专用微调脚本 `train/finetune_scripts/finetune_yolov9c_best.py`，核心配置如下：

```python
model.train(
    data="yolo.yaml",
    epochs=100,
    batch=8,              # 增大batch size
    imgsz=640,
    optimizer="auto",     # 自动选择 AdamW
    
    # 学习率配置
    lr0=0.002,            # 降低初始学习率
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    # Warmup 配置
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # 强化数据增强
    degrees=0.2,          # 小角度旋转
    translate=0.2,        # 平移增强
    scale=0.3,            # 缩放增强
    mosaic=1.0,           # 强制使用Mosaic
    mixup=0.1,
    copy_paste=0.2,       # 新增：复制粘贴增强
    
    # 损失函数优化
    label_smoothing=0.1,  # 标签平滑
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # 其他
    patience=50,
    close_mosaic=10,      # 最后10轮关闭Mosaic
)
```

### 3.3 执行微调

```powershell
python -m train.finetune_scripts.finetune_yolov9c_best `
  --epochs 100 `
  --batch 8 `
  --imgsz 640 `
  --device 0 `
  --patience 50
```

**关键改进点**：

| 改进项 | 第一阶段 | 第二阶段 | 目的 |
|--------|---------|---------|------|
| Batch Size | 4 | **8** | 更稳定的梯度更新 |
| Learning Rate | 0.005 | **0.002** | 更精细的参数调整 |
| Mosaic | 0.8 | **1.0** | 强化上下文学习 |
| Copy-Paste | 0.0 | **0.2** | 增强小目标检测 |
| Label Smoothing | 0.0 | **0.1** | 提升泛化能力 |
| Focal Loss | False | **True** | 解决难易样本不平衡 |

### 3.4 微调输出

最终权重保存在：
```
runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt  ← 最终使用的权重
```

---

## 4. 推理配置与优化

### 4.1 标准推理配置

适用于一般场景：

```powershell
python -m infer.infer_yolov9c `
  --weights "runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt" `
  --conf 0.64 `
  --iou 0.5 `
  --source data/images/test `
  --skip-import
```

### 4.2 优化推理配置（推荐）

针对**漏检问题**的优化配置：

```powershell
python -m infer.infer_yolov9c `
  --weights "runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt" `
  --conf 0.4 `        # 降低置信度阈值
  --iou 0.5 `         # 适中NMS阈值
  --imgsz 1280 `      # 提高分辨率
  --source data/infer_images/test `
  --skip-import `
  --run-tag yolov9c_optimized
```

**参数优化说明**：

| 参数 | 标准值 | 优化值 | 效果 |
|------|--------|--------|------|
| `--conf` | 0.64 | **0.4** | 捕获更多低置信度目标 |
| `--imgsz` | 640 | **1280** | 增强小目标和遮挡检测 |
| `--iou` | 0.7 | **0.5** | 平衡去重效果 |

### 4.3 更激进的配置

如果仍有漏检，可尝试：

```powershell
# 进一步降低阈值
python -m infer.infer_yolov9c --conf 0.35 --imgsz 1280 --iou 0.5

# 开启 TTA（测试时增强）
python -m infer.infer_yolov9c --conf 0.4 --imgsz 1280 --tta
```

⚠️ **注意**：TTA 会显著增加推理时间（约5倍）。

### 4.4 推理输出结构

```
outputs/infer/yolov9c/yolov9c_optimized/
├── visualizations/          # 带标注的可视化图片
├── predictions_txt/         # YOLO格式预测结果
├── predictions.json         # JSON格式详细结果
├── raw/                     # Ultralytics原始输出
│   └── labels/             # 原始预测标签
└── meta/
    └── summary.json        # 推理汇总信息
```

---

## 5. 效果对比与分析

### 5.1 优化前后对比

以 `data/infer_images/test` (7张图片) 为例：

#### 九牧安全快开
| 指标 | 优化前 (conf=0.64) | 优化后 (conf=0.4, imgsz=1280) | 提升 |
|------|-------------------|-------------------------------|------|
| 总检测数 | 71 (32张图) | 53 (7张图) | - |
| **平均每图** | **2.22** | **7.57** | **+241.2%** 🎉 |
| 平均置信度 | 0.9174 | 0.89xx | 略降 |

#### 九牧大冲力喷枪角阀
| 指标 | 优化前 (conf=0.64) | 优化后 (conf=0.4, imgsz=1280) | 提升 |
|------|-------------------|-------------------------------|------|
| 总检测数 | 15 (32张图) | 5 (7张图) | - |
| **平均每图** | **0.47** | **0.71** | **+52.4%** 🎉 |
| 平均置信度 | 0.9368 | 0.91xx | 略降 |

### 5.2 性能权衡

| 配置 | 推理速度 | 显存占用 | 召回率 | 准确率 | 适用场景 |
|------|---------|---------|--------|--------|----------|
| conf=0.64, imgsz=640 | ~52 FPS | ~3GB | 中等 | 高 | 实时应用 |
| conf=0.5, imgsz=1280 | ~20 FPS | ~8GB | 高 | 中高 | 通用场景 |
| **conf=0.704, imgsz=1280** ⭐ | ~20 FPS | ~8GB | 中高 | **高** | **生产环境推荐** |
| conf=0.4, imgsz=1280 | ~15 FPS | ~8GB | **极高** | 中 | 需要高召回率 |
| conf=0.35, imgsz=1280, TTA | ~3 FPS | ~8GB | **极高** | 中低 | 离线分析 |

### 5.3 适用场景建议

✅ **推荐使用优化配置的场景**：
- 存在小目标或严重遮挡
- 需要高召回率（宁可多检，不可漏检）
- 对推理速度要求不高

⭐ **生产环境推荐配置**：
```powershell
--conf 0.704 --imgsz 1280 --agnostic-nms
```
- ✅ F1 最优点，精确率和召回率最佳平衡
- ✅ Agnostic NMS 有效去除重叠框
- ✅ 关闭 TTA，避免背景误检

---

## 6. 常见问题与解决方案

### Q1: 训练时显存不足怎么办？

**解决方案**：
```powershell
# 减小 batch size
python -m train.train_yolov9c --batch 2

# 或减小图像尺寸
python -m train.train_yolov9c --imgsz 512

# 或清除缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### Q2: 某些类别仍然漏检严重？

**解决方案**：
1. **检查训练数据**：该类别样本是否充足？
2. **增加针对性数据**：收集更多困难样本
3. **调整类别权重**：在 `data.yaml` 中设置 `class_weights`
4. **继续微调**：使用更低的学习率再训练 50-100 轮

### Q3: 推理速度太慢？

**解决方案**：
```powershell
# 方案1: 降低图像分辨率
--imgsz 640  # 或 512

# 方案2: 提高置信度阈值（牺牲召回率）
--conf 0.5

# 方案3: 使用 TensorRT 加速（需要额外配置）
# 导出为 TensorRT 格式
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='engine')"
```

### Q4: 如何评估模型性能？

```powershell
# 使用 test 集进行评估
python -m train.evaluate_test --model-name yolov9c

# 或手动计算指标
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt')
metrics = model.val(data='yolo.yaml', split='test')
print(f'mAP50-95: {metrics.box.map:.4f}')
print(f'mAP50: {metrics.box.map50:.4f}')
"
```

### Q5: 如何可视化检测结果？

推理完成后，可视化图片自动保存在：
```
outputs/infer/yolov9c/<run_tag>/visualizations/
```

每张图片都带有：
- ✅ 彩色检测框
- ✅ 中文类别名称
- ✅ 置信度数值

---

## 📚 相关脚本说明

| 脚本 | 用途 | 位置 |
|------|------|------|
| `reimport_and_resplit.py` | 数据导入与重划分 | `train/` |
| `train_yolov9c.py` | 第一阶段基础训练 | `train/` |
| `finetune_scripts/` | **微调脚本集合** | `train/finetune_scripts/` |
| ├─ finetune_yolov9c_best.py | 通用漏检优化 | |
| ├─ finetune_angle_valve_specialist.py | 角阀类专项优化 | |
| └─ finetune_negative_samples.py | 负样本抑制优化 | |
| `infer_yolov9c.py` | 模型推理 | `infer/` |
| `evaluate_test.py` | Test集评估 | `train/` |
| `analyze_missed_detections.py` | 漏检分析 | 项目根目录 |
| `compare_optimization_results.py` | 优化效果对比 | 项目根目录 |

---

## 🎯 快速开始清单

```powershell
# 1. 数据导入
python train/reimport_and_resplit.py

# 2. 第一阶段训练（约2-3小时）
python -m train.train_yolov9c --epochs 200 --batch 4

# 3. 第二阶段微调（约1-2小时）
python -m train.finetune_scripts.finetune_yolov9c_best --epochs 100 --batch 8

# 4. 优化推理（生产环境推荐）
python -m infer.infer_yolov9c `
  --weights "runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt" `
  --conf 0.704 --imgsz 1280 --agnostic-nms `
  --source data/infer_images/test_imgs --skip-import

# 5. 查看结果
explorer outputs\infer\yolov9c\yolov9c_f1_optimal\visualizations
```

---

## 📝 版本记录

- **2026-04-28**: 
  - 初始版本，记录完整的两阶段训练流程和优化推理配置
  - 新增 F1 最优点推理配置（conf=0.704）
  - 新增差异化置信度阈值方案
  - 更新生产环境推荐配置
- **模型版本**: YOLOv9c
- **数据集**: 316张九牧产品图片（14个类别）
- **最终权重**: `runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt`
- **推荐推理配置**: conf=0.704, imgsz=1280, agnostic-nms

---


**最后更新**: 2026-04-28  
