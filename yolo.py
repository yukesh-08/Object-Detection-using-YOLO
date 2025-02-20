import torch
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "yolov8s.pt"
model = YOLO(model_path)

def real_time_detection():
    """Perform real-time object detection using YOLOv8 with POCO dataset."""
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{int(cls)}: {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Real-Time Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

poco_dataset_path = "path/to/poco_dataset/"

def train_on_poco():
    """Train YOLOv8 on the POCO dataset."""
    model.train(data=poco_dataset_path, epochs=50, imgsz=640)

real_time_detection()
