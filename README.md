# 九牧货架商品检测训练项目

## 1. 目录结构

```text
jomoo-testmodel/
├─ data/                       # 数据
├─ outputs/                    # 训练归档结果
├─ runs/                       # Ultralytics 原始输出
├─ train/
│  ├─ common.py                # 共享训练底座
│  ├─ train_all.py             # 一键训练 YOLOv8s + YOLOv9c + YOLOv11s
│  ├─ train_yolov8s.py         # 只训练 YOLOv8s
│  ├─ train_yolov9c.py         # 只训练 YOLOv9c
│  ├─ train_yolov11s.py        # 只训练 YOLOv11s
│  └─ evaluate_test.py         # 训练后用 test 集做最终评估
├─ infer/
│  ├─ common.py                # 共享推理底座
│  ├─ infer_all.py             # 一键推理 YOLOv8s + YOLOv9c + YOLOv11s
│  ├─ infer_yolov8s.py         # 只推理 YOLOv8s
│  ├─ infer_yolov9c.py         # 只推理 YOLOv9c
│  └─ infer_yolov11s.py        # 只推理 YOLOv11s
├─ finetune_negative_samples.py # 负样本微调训练脚本
├─ data.yaml                   # 数据集配置
├─ yolo.yaml                   # 默认数据集配置
├─ requirements.txt
└─ main.py
```

## 2. 数据准备

将数据按 YOLO 检测格式放到 `data/`：

```text
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

标注格式：`class x_center y_center width height`（归一化到 0~1）。

默认类别在 `train/common.py` 的 `CLASS_NAMES` 中，已按九牧 14 类配置好。

训练默认只使用 `train`，验证使用 `val`；`test` 作为最终留出集，在训练后单独评估。

## 3. 安装依赖

```powershell
pip install -r requirements.txt
```

## 4. 开始训练

一键训练三个模型（推荐）：

```powershell
python -m train.train_all
```

分别训练：

```powershell
python -m train.train_yolov8s
python -m train.train_yolov9c
python -m train.train_yolov11s
```

如果数据不在默认 `data/`，可指定路径：

```powershell
python -m train.train_all --data-root E:\path\to\your\data
```

## 5. 默认训练参数

当前三个训练入口统一使用同一套基线参数：

- `imgsz=640`
- `batch=4`
- `workers=4`
- `device=0`
- `epochs=200`
- `patience=30`
- `optimizer=AdamW`
- `cos_lr=True`
- `lr0=0.001`
- `weight_decay=0.001`
- `close_mosaic=20`
- `hsv_h=0.015`
- `hsv_s=0.6`
- `hsv_v=0.4`
- `degrees=10`
- `scale=0.5`
- `translate=0.15`
- `mosaic=0.8`
- `mixup=0.15`
- `fliplr=0.5`

同时开启训练批次级别随机增强：亮度、饱和度、噪点、轻微 gamma 变化。

此外，训练底座内置了早停机制：当验证集指标在 `patience` 个 epoch 内不再提升时，会自动停止训练。`train.train_all` 支持用 `--patience` 显式覆盖模型默认值，例如：

```powershell
python -m train.train_all --patience 10
```

## 6. 训练结果保存位置

每次训练会同时保留两类输出：

- Ultralytics 原始目录：`runs/jomoo/<run_name>`
- 标准归档目录：`outputs/<model_name>/<run_tag>/`

标准归档目录结构：

```text
outputs/
  yolov8s/
    20260420_120000_yolov8s/
      weights/                # best.pt, last.pt
      visualizations/         # results.png, PR 曲线、混淆矩阵等图
      logs/                   # results.csv, args.yaml
      meta/                   # summary.json（训练参数和索引信息）
```

> `run_tag` 默认时间戳，可通过 `--run-tag` 自定义。

## 7. 常用可选参数

```powershell
python -m train.train_all --run-tag exp_001
python -m train.train_all --output-root E:\exp_outputs
python -m train.train_yolov9c --batch 8 --device 0
```

## 8. 注意事项

- `yolov9c.pt` 与 `yolo11s.pt` 需要当前 `ultralytics` 版本支持对应模型。
- 如果显存不足，先降低 `--batch` 或改小 `--imgsz`。
- `yolo.yaml` 若不存在会自动生成；已存在则直接使用。

## 9. 批量推理

默认会从以下目录导入图片后再推理：

- `E:\code\jiumu_product_recognition\data\test_imgs`

一键推理三个模型：

```powershell
python -m infer.infer_all
```

开启 TTA 推理：

```powershell
python -m infer.infer_all --tta
```

默认权重优先级：

1. 命令行显式传入的 `--weights-v8/--weights-v9/--weights-v11`
2. `outputs/<model_name>/` 下最新训练产物中的 `weights/best.pt`
3. 根目录预训练权重 `yolov8s.pt` / `yolov9c.pt` / `yolo11s.pt`

也可以显式指定来源目录：

```powershell
python -m infer.infer_all --source E:\code\jiumu_product_recognition\data\test_imgs
```

若不想导入到项目内，直接对原目录推理：

```powershell
python -m infer.infer_all --skip-import
```

如果你的训练产物不在默认 `outputs/`，可以显式指定训练产物根目录：

```powershell
python -m infer.infer_all --trained-root E:\path\to\outputs
```

分别跑单模型：

```powershell
python -m infer.infer_yolov8s
python -m infer.infer_yolov9c
python -m infer.infer_yolov11s
```

单模型同样可以开启 TTA，例如：

```powershell
python -m infer.infer_yolov8s --tta
```

### 10. 使用 test 集做最终评估

训练结束后，如果你想把 `test` 集真正用起来，可以直接对最新训练权重做最终评估：

```powershell
python -m train.evaluate_test
```

默认会依次评估 `yolov8s`、`yolov9c`、`yolov11s` 的最新 `best.pt`，并使用 `split=test`。

如果只想评估某一个模型：

```powershell
python -m train.evaluate_test --model-name yolov9c
```

### 11. 优化推理配置（针对漏检问题）

对于容易漏检的类别（如**九牧安全快开**、**九牧大冲力喷枪角阀**），推荐使用以下优化参数：

#### 推荐配置

```powershell
# 降低置信度阈值 + 提高图像分辨率
python -m infer.infer_yolov9c `
  --weights "runs/detect/runs/jomoo/yolov9c_jomoo_finetune/weights/best.pt" `
  --conf 0.4 `
  --iou 0.5 `
  --imgsz 1280 `
  --source data/infer_images/test `
  --skip-import
```

#### 参数说明

| 参数 | 默认值 | 优化值 | 说明 |
|------|--------|--------|------|
| `--conf` | 0.64 | **0.4** | 降低置信度阈值，捕获更多低置信度目标 |
| `--imgsz` | 640 | **1280** | 提高分辨率，增强小目标和遮挡场景检测 |
| `--iou` | 0.7 | **0.5** | 适中的NMS阈值，平衡去重效果 |

#### 优化效果

以 `data/infer_images/test` (7张图片) 为例：

- **九牧安全快开**: 平均每图检测数从 2.22 → 7.57 (**+241.2%**)
- **九牧大冲力喷枪角阀**: 平均每图检测数从 0.47 → 0.71 (**+52.4%**)

#### 适用场景

✅ **推荐使用优化配置的场景：**
- 存在小目标或遮挡严重的情况
- 需要更高的召回率（宁可多检，不可漏检）
- 对推理速度要求不高（imgsz=1280 会增加推理时间）

⚠️ **注意事项：**
- 降低 `conf` 可能增加误检，需根据实际业务权衡
- 增大 `imgsz` 会显著增加显存占用和推理时间
- 建议在验证集上测试不同参数组合，找到最佳平衡点

#### 其他可选优化

```powershell
# 更激进的配置（进一步降低阈值）
python -m infer.infer_yolov9c --conf 0.35 --imgsz 1280 --iou 0.5

# 开启 TTA 测试时增强（进一步提升准确率）
python -m infer.infer_yolov9c --conf 0.4 --imgsz 1280 --tta
```

### 推理输出目录

每次推理会保存到：`outputs/infer/<model_name>/<run_tag>/`

```text
outputs/
  infer/
    yolov8s/
      20260421_120000_yolov8s/
        visualizations/         # 带框可视化图（含 labels/）
        predictions_txt/        # YOLO txt 预测结果（拷贝）
        predictions.json        # 全量预测结果 JSON
        meta/
          summary.json          # 本次推理汇总信息
```

说明：

- `data/infer_images/test_imgs/` 为导入后的项目内图片目录（默认每次重建）。
- 支持图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。
- 中文类别文字会自动尝试使用 Windows 系统中文字体（如 `Microsoft YaHei` / `SimHei`）；若系统没有可用字体，请补装中文字体。
- 推理前会根据 `data.yaml` 的 `names` 自动更新 `data/labels/*/classes.txt`。
- 若要确保使用最新训练权重，优先检查 `outputs/<model_name>/<run_tag>/weights/best.pt` 是否存在。
- `--tta` 会把 `augment=True` 传给 Ultralytics `predict()`，默认关闭。

