from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model("../groceries.jpg")