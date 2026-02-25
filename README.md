# 🎓 Smart Student Activity Monitor (Backend)

An AI-powered monitoring system that uses **ESP32-CAM** hardware and **YOLOv8** computer vision to analyze student activity in real-time. This project provides a live streaming dashboard with automated detection of participation, distractions, and unusual behaviors.

## 🌟 Features

* **Real-Time AI Analysis:** Leverages YOLOv8 for object detection and pose estimation.
* **Activity Recognition:** Automatically detects:
* Writing/Focused work
* Hand raised (Participation)
* Sleeping/Distracted (Head down detection)
* Using phone (Unusual activity)
* Student absence


* **Multi-Camera Live Stream:** Supports multiple concurrent camera feeds via a modern web dashboard.
* **Dual Mode Detection:** * *Advanced:* Uses Ultralytics YOLOv8 for high-accuracy pose/object detection.
* *Basic:* Fallback to OpenCV Haar Cascades and motion density if hardware resources are limited.


* **Data Privacy:** Only analysis metadata is stored on disk (JSON); live video frames are kept in-memory only.

---

## 🛠️ System Architecture

1. **Edge Device:** ESP32-CAM captures images and sends them via POST requests to the Flask API.
2. **Backend:** Flask server processes the images using a threaded architecture to handle simultaneous analysis and streaming.
3. **AI Engine:** YOLOv8 (Object) and YOLOv8-Pose models analyze student posture and classroom objects.
4. **Frontend:** A responsive HTML5/JavaScript dashboard with real-time stats and activity logs.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* OpenCV
* Ultralytics (YOLOv8)
* Flask & Flask-CORS

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/student-activity-monitor.git
cd student-activity-monitor

```


2. **Install dependencies:**
```bash
pip install flask flask-cors opencv-python numpy ultralytics

```


3. **Download Models:**
The system will automatically download `yolov8n.pt` and `yolov8n-pose.pt` on the first run.
4. **Run the Server:**
```bash
python app.py

```



---

## 📡 Hardware Integration (ESP32-CAM)

To connect your ESP32-CAM, configure your Arduino sketch to send a multipart POST request to:
`http://<YOUR_SERVER_IP>:5000/api/analyze`

**Payload Requirements:**

* `image`: The JPEG buffer.
* `student_id`: A unique string identifier for the student/camera.

---

## 📊 API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Access the web-based dashboard. |
| `/api/analyze` | POST | Receive and analyze an image from the camera. |
| `/api/stream/<id>` | GET | MJPEG Video stream for a specific Student ID. |
| `/api/stats` | GET | Retrieve overall activity statistics. |
| `/api/sessions` | GET | Get the recent history of detected activities. |

---

## 🖼️ Dashboard Preview

The dashboard includes a **Statistics Grid** for quick oversight and a **Live Stream Tab** for visual monitoring.

## ⚙️ Configuration

Analysis data is stored in the `analysis_data/` directory:

* `sessions.json`: Historical log of detected activities.
* `statistics.json`: Aggregated counts and distribution data.

---


## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
