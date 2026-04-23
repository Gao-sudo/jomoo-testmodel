def main() -> None:
    print("请在项目根目录执行以下脚本之一：")
    print("  python -m train.train_all")
    print("  python -m train.train_yolov8s")
    print("  python -m train.train_yolov9c")
    print("  python -m train.train_yolov11s")
    print("  python -m infer.infer_all")
    print("  python -m infer.infer_yolov8s")
    print("  python -m infer.infer_yolov9c")
    print("  python -m infer.infer_yolov11s")

if __name__ == "__main__":
    main()
