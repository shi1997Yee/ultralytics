"""
准备：
python环境配置：参考docs/guides/conda-quickstart.md
数据集制作，参考docs/datasets/下的指导制作自己的数据集
修改配置文件：
    modifiy ultralytics/cfg/datasets/*.yaml for your own dataset
    modify ultralytics/cfg/model/v8/yolov8-cls.yaml , nc: number of classes
训练例子：参考docs/usage/python.md  or docs/tasks/classify.md
"""

from ultralytics import YOLO
from importlib.metadata import PackageNotFoundError, version
# Load a model
model = YOLO('/Users/shitiantian05/sttCodes/test0818/ultralytics/yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model  device='mps'支持苹果M1M2芯片
results = model.train(data='/Users/shitiantian05/sttCodes/test0818/ultralytics/mydataset', epochs=20, imgsz=640, device='mps')