# Traffic Signal Violation Detection System

> **Implementation of the Research Paper:**  
> *"Traffic Signal Violation Detection using Artificial Intelligence and Deep Learning"*  
> **Authors:** Dr. S. Raj Anand, Dr. Naveen Kilari, Dr. D. Udaya Suriya Raj Kumar  
> **Publication:** *International Journal of Advanced Research in Engineering and Technology (IJARET)*, Volume 12, Issue 2, February 2021, pp. 207-217.

An end-to-end Computer Vision and Deep Learning system designed for automated traffic surveillance. It detects multiple traffic violations simultaneously in real-time video footage:
- 🚦 **Traffic Signal Jump Detection**: Monitors designated stop lines and flags vehicles crossing during a **RED** light.
- ⚡ **Speed Estimation & Speed Violation**: Computes frame-by-frame velocity using spatial displacement of road-contact anchor points and camera calibration, flagging over-speeding vehicles.
- 🚗 **Vehicle Counting & Classification**: Tracks and categorizes vehicles (cars, motorcycles, buses, trucks) passing through the scene.
- 📸 **Automated Evidence Capture**: Automatically crops high-resolution snapshots of violating vehicles with metadata banners (ID, timestamp, speed, violation type) and exports audit logs (CSV/JSON).
- 💻 **Interactive Web Dashboard**: Live surveillance stream with drag-and-drop stop line placement, signal control remote, and evidence gallery.

---

## 🚀 Quickstart & Developer Setup Guide

Follow the industry-standard workflow below to set up and run this project in a clean, isolated virtual environment.

### 1. Prerequisites
- **Python 3.10, 3.11, 3.12, or 3.13** installed on your system.
- **Git** (optional, for cloning).

---

### 2. Create a Virtual Environment

Open your terminal in the project root directory (`d:/Projects/DeepLearning`) and create a Python virtual environment:

**On Windows (PowerShell / Command Prompt):**
```powershell
# Create the virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# In PowerShell:
.\venv\Scripts\Activate.ps1

# Or in Command Prompt (cmd):
venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

---

### 3. Install Project Dependencies

Upgrade `pip` and install all required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Launch the Interactive Web Dashboard (Recommended)

Start the Flask surveillance server:
```bash
python app.py
```
Once started, open your web browser and navigate to:
👉 **`http://127.0.0.1:5000/`**

**Dashboard Features:**
- **Live Stream Preview**: Real-time bounding boxes (Green = compliant, Red = violation), speed badges, and HUD telemetry.
- **Interactive Stop Line**: Click and drag directly across any lane on the video player to reposition the stop line in real-time.
- **Traffic Signal Remote**: Switch between `Auto Cycle`, `Green`, `Yellow`, or `Red (Enforce)`.
- **Custom Video Selector**: Choose between included CCTV footage (`traffic_ip_camera.mp4`, `roads.mp4`, or `synthetic_demo.mp4`).
- **Telemetry Sliders**: Adjust speed limit (km/h), detection confidence threshold, and calibration scale on the fly.
- **Evidence Modal**: Click any card in the incident feed to inspect the full-resolution violation snapshot and download the evidence image.
- **Export Report**: Download the full incident log as a formatted CSV file anytime.

---

### Option B: Headless Batch Processing via CLI

To process a video file from the command line and export an annotated output video:

```bash
# Process CCTV surveillance video
python main.py --video videos/traffic_ip_camera.mp4 --stop-line "700,250,1200,250" --speed-limit 45 --output output_cctv.mp4

# Run with manual RED signal enforcement
python main.py --video videos/traffic_ip_camera.mp4 --stop-line "700,250,1200,250" --signal-mode MANUAL --initial-state RED

# Process with live OpenCV window display
python main.py --video videos/traffic_ip_camera.mp4 --show
```

**Available CLI Flags:**
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--video` | `videos/traffic_ip_camera.mp4` | Path to input video file |
| `--output` | `output_annotated.mp4` | Path to save annotated video |
| `--stop-line` | `700,250,1200,250` | Stop line coordinates `x1,y1,x2,y2` |
| `--speed-limit` | `50.0` | Speed violation threshold (km/h) |
| `--signal-mode` | `AUTO_CYCLE` | Traffic signal mode: `AUTO_CYCLE`, `MANUAL`, `COLOR_ROI` |
| `--initial-state`| `GREEN` | Initial signal state: `GREEN`, `YELLOW`, `RED` |
| `--max-frames` | `0` (all) | Maximum number of frames to process |
| `--show` | `False` | Display live OpenCV GUI playback window |

---

### Option C: Generate Synthetic Traffic Video

You can synthesize custom traffic scenarios with road lanes, traffic signals, and simulated vehicles:

```bash
python generate_sample_video.py
```
This generates `videos/synthetic_demo.mp4` containing compliant cars, red-light runners, and speeders for immediate testing.

---

## 📂 Project Architecture

```
DeepLearning/
├── app.py                     # Flask web backend with MJPEG streaming & REST APIs
├── detector.py                # YOLOv8 object detector for vehicle classification
├── tracker.py                 # Multi-object tracker with road contact anchor points
├── speed_estimator.py         # Perspective-aware velocity calculation engine
├── signal_detector.py         # Traffic signal phase controller (Auto, Manual, HSV ROI)
├── violation_detector.py      # Stop-line intersection math & evidence snapshot generator
├── annotator.py               # Visual overlay renderer (Green=Normal, Red=Violation)
├── pipeline.py                # Unified coordinator binding all detection & violation modules
├── main.py                    # Standalone CLI batch processor with progress bar & reporting
├── generate_sample_video.py   # Synthetic road intersection video generator
├── requirements.txt           # Python dependency specification
├── README.md                  # Project documentation & setup instructions
├── templates/
│   └── index.html             # Web dashboard frontend template
├── static/
│   ├── style.css              # Modern glassmorphic dark-theme styles
│   └── app.js                 # Interactive frontend logic & canvas stop-line dragging
├── videos/                    # Input video datasets (CCTV, roads, synthetic demo)
└── violations/                # Exported violation evidence snapshot images
```

---

## 📊 Output Artifacts & Reports

Every time a violation occurs, the system records:
1. **Evidence Snapshots (`violations/violation_<id>_<class>.jpg`)**: A high-resolution crop of the offending vehicle with a metadata banner showing Vehicle ID, Speed, Timestamp, and Offense category.
2. **Audit CSV Report (`violations_report.csv`)**: Tabular export of all infractions for administrative review.
3. **Audit JSON Report (`violations_report.json`)**: Machine-readable telemetry report containing summary metrics and per-vehicle trajectories.
