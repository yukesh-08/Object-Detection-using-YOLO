import numpy as np
import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLO model
model_path = "yolov8s.pt"
model = YOLO(model_path)

poco_dataset_path = "path/to/poco_dataset/"  # Update with actual path

def real_time_detection():
    """Perform real-time object detection using YOLOv8 with optional face recognition."""
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
                
                # Face Recognition (Optional)
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    try:
                        result = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)
                        age = result[0]['age']
                        gender = result[0]['dominant_gender']
                        cv2.putText(frame, f"{gender}, {age}", (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    except Exception as e:
                        print("Face analysis failed:", e)
        
        cv2.imshow("Real-Time Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def train_on_poco():
    """Train YOLOv8 on the POCO dataset."""
    model.train(data=poco_dataset_path, epochs=50, imgsz=640)

if __name__ == "__main__":
    real_time_detection()
