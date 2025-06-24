# Intelligent Face Tracker with Auto-Registration and Visitor Counting

## 🚀 Overview

This project is a real-time AI-based intelligent visitor tracking system designed to detect, track, and recognize faces from a video stream. It automatically registers new faces, identifies them across frames, tracks them until they exit the frame, and logs all events (entry/exit) with cropped images and metadata.

---

## 🎯 Objective

- Detect faces using YOLOv8
- Recognize and generate embeddings using InsightFace
- Track faces frame-to-frame using OpenCV + Kalman Filter
- Log each **entry** and **exit** event with:
  - Cropped face image
  - Timestamp
  - Unique Face ID
- Maintain a count of **unique visitors**
- Store all logs and metadata in a **SQLite database** and local folder structure

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
│           ├── det_10g.onnx
│           ├── genderage.onnx
│           └── w600k_r50.onnx       # Downloaded manually
├── videoDatasets/
│   └── input.mp4             # Not tracked in Git
├── database/
│   └── faces.db              # Not tracked in Git
├── logs/
│   ├── system.log
│   ├── events.log
│   └── entries/
│       └── YYYY-MM-DD/
│           └── *.jpg
├── requirements.txt
└── README.md
```

> **Note:** Large files (like `.onnx`, `.pt`, `.mp4`, `.db`, `.log`, `.jpg`) are **not included** in the repository but required for execution.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:sameers07/face_tracker.git
cd face_tracker
```

---

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download Required YOLOv8 Model

The YOLOv8n-face model is **not included** in the repository due to size limits. You must download it manually:

- ✅ Download `yolov8n-face.pt` from the [Ultralytics release page](https://github.com/ultralytics/ultralytics/releases).
- 📁 Place it into the `models/` directory:

```bash
models/yolov8n-face.pt
```

Make sure this path matches what's configured in your `config/config.json`.

---

## ⚙️ Sample `config.json`

```json
{
   "detection": {
        "model": "../models/yolov8n-face.pt",
        "model_path": "../models",
        "skip_frames": 1,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.4,
        "min_face_size": 40
    },
    "recognition": {
        "model": "buffalo_l",
        "model_path": "../models/insightface",
        "gpu_id": -1,
        "input_size": [320, 320],
        "similarity_threshold": 0.6,
        "min_face_size": 40,
        "batch_size": 8
    },
    "tracking": {
        "max_age": 30,
        "appearance_weight": 0.7,
        "reid_threshold": 0.5,
        "kalman_filter": {
            "enabled": true,
            "process_noise": 0.1,
            "measurement_noise": 1.0
        }
    },
    "video": {
        "input_path": "../videoDatasets/input.mp4",
        "output_path": "../logs/output.avi",
        "max_duration": 300,
        "output_codec": "mp4v",
        "output_fps":15
    },
    "rtsp": {
        "enabled": false,
        "url": "rtsp://example.com/stream",
        "max_retries": 5,
        "timeout": 10
    },
    "database": {
        "path": "../database/faces.db",
        "auto_cleanup_days": 30,
        "backup_interval_hours": 24
    },
    "logging": {
        "log_dir": "../logs",
        "log_level": "INFO",
        "max_log_size_mb": 100,
        "retention_days": 30,
        "console_output": true
    },
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
[Kalman-Based Face Tracking]
      ↓
[Event Logger]
   ├── Crops Face Images
   ├── Logs Entry/Exit Events
   └── Updates DB + File System
```

---

## 📦 Features

- Real-time face detection using YOLOv8
- Embedding-based recognition with InsightFace
- Kalman filter–assisted tracking
- SQLite-based logging
- Frame-skipping for performance
- Entry/Exit tracking with metadata
- Clean folder structure and modular code

---

## 🧪 Testing Instructions

```bash
python3 src/main.py
```

To test with RTSP stream (update in config first):

```bash
cv2.VideoCapture("rtsp://<your-camera-url>")
```

---

## 📽️ Demo Video

🎬 [Add your Loom or YouTube link here]

---

## ✅ Assumptions

- Camera frames must show clear, front-facing faces
- Similarity threshold and skip_frames should be tuned per hardware
- Only `.onnx` models compatible with InsightFace are supported

---

## 🔧 Future Improvements

- Web dashboard for monitoring visitors in real-time
- Email/SMS alerts on repeated visitor entry
- Cloud database & image syncing
- Face clustering + duplicate prevention

---

## 📌 Submission Checklist

✅ Source code  
✅ Clean folder structure  
✅ README file with setup  
✅ GitHub-ready (no large files)  
✅ `.gitignore` for safe sharing  
✅ Local DB and logs preserved by `.gitkeep` only

---

**This project is part of a hackathon run by [Katomaran](https://katomaran.com)**
