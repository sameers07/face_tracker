from ultralytics import YOLO
import cv2
import os
print("[DEBUG] CWD:", os.getcwd())


# Load YOLOv8 face model
model = YOLO('../models/yolov8n-face.pt')  # make sure this path is correct

# Load test image (place one face image in the root dir and rename it to sample.jpg)
img = cv2.imread('sample.jpg')
if img is None:
    print("Image not found! Please add a sample.jpg image.")
    exit()

# Run detection
results = model(img)

# Draw and show results
for result in results:
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Face Detection", annotated_frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

