# CCPD2019 车牌检测与识别

本项目用 Ultralytics YOLO 定位车牌，再用 CRNN + CTC 识别省份、字母和数字。网页界面支持 CCPD2019 预置图片和自定义上传。

> **模型记录**：仓库附带的检测权重 `runs/yolo_plate/weights/best.pt` 的训练记录为 **yolo11n.pt**，不是 YOLO 26。`configs/project.yaml` 为新的训练运行指定了 YOLO 26，并在权重不可用时回退到 YOLO11。请按实际训练记录报告模型版本。

## 仓库内容

- `src/ccpd.py`：解析 CCPD2019 文件名中的框、四点和车牌字符。
- `scripts/prepare_yolo_data.py`、`scripts/train_yolo.py`：检测数据准备与训练。
- `scripts/prepare_crnn_data.py`、`src/crnn.py`、`scripts/train_crnn.py`：省份均衡抽样、CRNN 定义与训练。
- `scripts/infer_pipeline.py`：命令行检测加识别。
- `scripts/ui_app.py`、`web/index.html`：本地网页界面。
- `runs/yolo_plate/weights/best.pt`、`runs/crnn_plate_balanced/best.pt`：可直接用于推理的最佳权重。

原始 CCPD2019 图片和生成的训练数据不在仓库中。请从 [CCPD 项目](https://github.com/detectRecog/CCPD) 获取数据，放在本项目同级的 `CCPD2019` 目录，或在数据准备与 UI 命令中指定 `--dataset-root`。

## 环境与启动

建议使用 Python 3.11、支持 CUDA 的 PyTorch，以及 `requirements.txt` 中的依赖。Windows PowerShell 示例：

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/ui_app.py --device cuda --port 7860
```

打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)。Windows 用户也可以双击 `start_ui_gpu.bat`；它会等待服务就绪后打开浏览器。预置图片依赖 CCPD2019 数据目录，自定义上传可直接使用仓库中的模型权重。

如需 CPU 推理，把 `--device cuda` 改为 `--device cpu`。命令行推理示例：

```powershell
py -3.11 scripts/infer_pipeline.py --image "D:\path\to\vehicle.jpg"
```

结果默认保存到 `runs/prediction.jpg`。

## 重新训练

数据默认从项目同级的 `CCPD2019` 目录读取。数据位于其他位置时，为两个准备脚本分别传入 `--dataset-root`。

```powershell
py -3.11 scripts/prepare_yolo_data.py --max-images 4000
py -3.11 scripts/train_yolo.py --epochs 30 --imgsz 640 --batch 16 --device 0
py -3.11 scripts/prepare_crnn_data.py --max-per-province 800 --subsets ccpd_base ccpd_blur ccpd_challenge ccpd_db ccpd_fn ccpd_rotate ccpd_tilt ccpd_weather
py -3.11 scripts/train_crnn.py --epochs 20 --batch 128 --device cuda
```

CRNN 数据使用 CCPD 文件名里的四点坐标矫正为 160×32 裁剪图。原始 CCPD2019 的 `皖` 牌比例很高，因此数据准备默认按省份限额抽样。生成的数据保存在 `data/`，不会进入 Git。

## 当前训练结果

检测模型最后一轮的验证指标来自 `runs/yolo_plate/results.csv`：Precision 100.0%、Recall 99.984%、mAP50 99.5%、mAP50-95 77.718%。均衡 CRNN 权重保存的验证集整牌准确率约为 93.98%。这些是各自验证集指标，不代表任意上传图片的端到端准确率。

两份最佳权重随源码一起提交。数据集、训练缓存和用户上传图片由 `.gitignore` 排除。
