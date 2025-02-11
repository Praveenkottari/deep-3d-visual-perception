
from ultralytics import YOLO

weights = "./heads/yolov8n.pt"
classes = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck
#detetction model
def detection_model(weights=weights,classes=classes):
    model = YOLO(weights)
    model.classes = classes 
    return model

