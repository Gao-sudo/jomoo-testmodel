def main() -> None:
<<<<<<< HEAD
    print("请在项目根目录执行以下脚本之一：")
=======
    print("请直接运行以下脚本之一：")
>>>>>>> 06ec312e3a2859a28202aeb1c9ce0b884e3ba790
    print("  python -m train.train_all")
    print("  python -m train.train_yolov8s")
    print("  python -m train.train_yolov9c")
    print("  python -m train.train_yolov11s")
    print("  python -m infer.infer_all")
    print("  python -m infer.infer_yolov8s")
    print("  python -m infer.infer_yolov9c")
    print("  python -m infer.infer_yolov11s")
<<<<<<< HEAD
    print("\n提示：训练脚本会自动把相对路径解析到项目根，避免产物误落在 train/ 目录。")
=======
    print("\n共享训练/推理底座已抽离，不再把所有逻辑放在 `main.py` 里。")
>>>>>>> 06ec312e3a2859a28202aeb1c9ce0b884e3ba790


if __name__ == "__main__":
    main()
