def main() -> None:
    print("请直接运行以下脚本之一：")
    print("  python -m train.train_all")
    print("  python -m train.train_yolov8s")
    print("  python -m train.train_yolov9c")
    print("  python -m train.train_yolov11s")
    print("  python -m infer.infer_all")
    print("  python -m infer.infer_yolov8s")
    print("  python -m infer.infer_yolov9c")
    print("  python -m infer.infer_yolov11s")
    print("\n共享训练/推理底座已抽离，不再把所有逻辑放在 `main.py` 里。")


if __name__ == "__main__":
    main()
