import torch
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "/home/chentao/桌面/robot_nav/yolov5s.pt")  # or yolov5n - yolov5x6, custom

# Images
img = "/home/chentao/桌面/robot_nav/src/my_robot_navigation/scripts/2024-10-12 15-28-27屏幕截图.png"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img, size=320)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

# Post-processing
# Extracting information from results
detections = results.xyxy[0]  # tensor with shape (num_detections, 6)

for detection in detections:
    x1, y1, x2, y2, conf, cls = detection
    print(f"Bounding Box: ({x1}, {y1}), ({x2}, {y2})")
    print(f"Confidence: {conf}")
    print(f"Class: {cls}")

# Optionally, you can filter detections based on confidence threshold
confidence_threshold = 0.5
filtered_detections = [d for d in detections if d[4] >= confidence_threshold]

for detection in filtered_detections:
    x1, y1, x2, y2, conf, cls = detection
    print(f"Filtered Bounding Box: ({x1}, {y1}), ({x2}, {y2})")
    print(f"Filtered Confidence: {conf}")
    print(f"Filtered Class: {cls}")