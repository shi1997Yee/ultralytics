from ultralytics import YOLO
from importlib.metadata import PackageNotFoundError, version
# Load a model
model = YOLO('/Users/shitiantian05/sttCodes/test0818/ultralytics/yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/Users/shitiantian05/sttCodes/test0818/ultralytics/mydataset', epochs=100, imgsz=640, device='mps')