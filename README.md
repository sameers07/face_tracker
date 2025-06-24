# Intelligent Face Tracker with Auto-Registration and Visitor Counting

## 🚀 Overview

This project is a real-time AI-based intelligent visitor tracking system designed to detect, track, and recognize faces from a video stream. It automatically registers new faces, identifies them across frames, tracks them until they exit the frame, and logs all events (entry/exit) with cropped images and metadata.

---

## 🎯 Objective

- Detect faces using YOLOv8.
- Recognize and generate embeddings using InsightFace.
- Track faces frame-to-frame using OpenCV-based tracking.
- Log each **entry** and **exit** event with:
  - Cropped face image
  - Timestamp
  - Unique Face ID
- Maintain a count of **unique visitors**.
- Store all logs and metadata in a **SQLite database** and **local folder structure**.

---

## 📁 Project Structure

```
face_tracker/
├── config/
│   └── config.json
├── src/
│   ├── main.py
│   ├── detection.py
│   ├── recognition.py
│   ├── tracking.py
│   ├── database.py
│   └── custom_logger.py
├── models/
│   ├── yolov8n-face.pt
│   └── insightface/
│       └── buffalo_l/
│           ├── 1k3d68.onnx
│           ├── 2d106det.onnx
│           └── w600k_r50.onnx
├── videoDatasets/
│   └── input.mp4
├── database/
│   └── faces.db
├── logs/
│   ├── system.log
│   ├── events.log
│   └── entries/
│       └── YYYY-MM-DD/
│           └── *.jpg
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd face_tracker
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

Place the following models in the `models/` directory:
- `yolov8n-face.pt`
- InsightFace ONNX models: `w600k_r50.onnx`, `2d106det.onnx`, `1k3d68.onnx`

---

## ⚙️ Sample `config.json`

```json
{
  "frame_skip": 5,
  "recognition_threshold": 0.5
}
```

---

## 🧠 System Architecture

```
[Video Input]
      ↓
[YOLOv8 Face Detection]
      ↓
[InsightFace Recognition] ↔ [SQLite DB]
      ↓
[OpenCV Tracking]
      ↓
[Logging Module]
   ├── Logs Events
   ├── Crops and Saves Face Images
   └── Stores Metadata in DB
```

---

## 📦 Features

- Real-time face detection
- Face recognition with InsightFace
- Face tracking with OpenCV
- SQLite and filesystem logging
- Configurable detection frame skipping

---

## 📊 Output Example

- Cropped images: `logs/entries/YYYY-MM-DD/`
- Log files: `system.log`, `events.log`
- DB: `faces.db` for metadata

---

## 🧪 Testing Instructions

```bash
python src/main.py
```

To use RTSP camera:

```python
cv2.VideoCapture("rtsp://your-camera-link")
```

---

## 📽️ Demo Video

👉 [Add your Loom or YouTube demo link here]

---

## ✅ Assumptions

- Clear frontal face visibility
- Minimal occlusion
- Threshold tuned for cosine similarity

---

## 🔧 Future Improvements

- Web UI
- Real-time monitoring dashboard
- Cloud sync for DB and logs

---

## 📌 Submission Checklist

✅ Code  
✅ README  
✅ Video demo  
✅ Logs & DB output  

---

**This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)**


---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

These libraries cover detection (YOLOv8), recognition (InsightFace), tracking (OpenCV), and general utilities.

---
