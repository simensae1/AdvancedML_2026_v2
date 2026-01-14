from ultralytics import YOLO
import os

# 1. Load a pre-trained "Nano" model (best for starting)
model = YOLO("yolo11x.pt")
dataset_path = os.path.abspath("pedestrian_Traffic_Light.v1i.yolov11/data.yaml")
# 2. Train the model
model.train(
    data=dataset_path,
    epochs=50,        # Number of passes through the data
    imgsz=640,        # Image resolution
    batch=8,
    device="0"        # Use "0" for NVIDIA GPU, or "cpu"
)


my_custom_pedestrian_model = YOLO("runs/detect/train2/weights/best.pt")
results = my_custom_pedestrian_model("PTL_Dataset_768x576/heon_IMG_0626.JPG")
results[0].save(filename="prediction_result.jpg")

model.save("my_custom_pedestrian_model.pt")
