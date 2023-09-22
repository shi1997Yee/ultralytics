"""
prepare
modifiy ultralytics/cfg/datasets/*.yaml for your own dataset
modify ultralytics/cfg/model/v8/yolov8-cls.yaml , nc: number of classes
"""

from ultralytics import YOLO
from importlib.metadata import PackageNotFoundError, version
# Load a model
model = YOLO('/Users/shitiantian05/sttCodes/test0818/ultralytics/yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/Users/shitiantian05/sttCodes/test0818/ultralytics/mydataset', epochs=20, imgsz=640, device='mps')