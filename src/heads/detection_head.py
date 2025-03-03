
from ultralytics import YOLO

weights = "/home/airl010/1_Thesis/deep-3d-visual-perception/weights/yolov8/yolov8n.pt"
classes = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck
#detetction model
def detection_model(weights=weights,classes=classes):
    model = YOLO(weights)
    model.classes = classes 
    return model

