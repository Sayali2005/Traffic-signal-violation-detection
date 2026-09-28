"""
Flask Web Application for Traffic Signal Violation Detection System.
Provides live MJPEG video streaming, interactive stop-line adjustment,
traffic light state controls, telemetry metrics, and violation evidence gallery.
"""

import os
import sys
import atexit
import signal
import time
import csv
import io
import json
import threading
from typing import Dict, Any, List, Optional
import cv2
from flask import Flask, render_template, Response, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from pipeline import TrafficViolationPipeline

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
VIOLATIONS_DIR = os.path.join(BASE_DIR, "violations")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

def clear_violations_dir():
    """Removes previous violation snapshots to keep the project lean and optimized."""
    if os.path.exists(VIOLATIONS_DIR):
        for f in os.listdir(VIOLATIONS_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    os.remove(os.path.join(VIOLATIONS_DIR, f))
                except Exception:
                    pass

# Automatically empty violations folder when stopping the code (Ctrl+C / termination)
atexit.register(clear_violations_dir)

def sig_exit_handler(sig, frame):
    clear_violations_dir()
    sys.exit(0)

try:
    signal.signal(signal.SIGINT, sig_exit_handler)
    signal.signal(signal.SIGTERM, sig_exit_handler)
except Exception:
    pass

# Always start with an empty violations folder
clear_violations_dir()

# Global Application State
class VideoStreamController:
    def __init__(self):
        clear_violations_dir()
        self.current_video_file = os.path.join(VIDEOS_DIR, "traffic_ip_camera.mp4")

        if not os.path.exists(self.current_video_file):
            vids = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]
            if vids:
                self.current_video_file = os.path.join(VIDEOS_DIR, vids[0])

        self.cap: Optional[cv2.VideoCapture] = None
        self.pipeline: Optional[TrafficViolationPipeline] = None
        self.lock = threading.Lock()
        self.is_running = True
        self.is_paused = False
        self.last_frame: Optional[bytes] = None
        self.current_stats: Dict[str, Any] = {}
        self.fps_actual = 0.0
        self.video_width = 1280
        self.video_height = 720
        self.stop_line = ((180, 500), (1120, 500))
        self.speed_limit = 50.0
        self.signal_mode = "AUTO_CYCLE"
        self.signal_state = "GREEN"
        self.mpp = 0.045
        
        self.init_pipeline()

    def init_pipeline(self):
        with self.lock:
            if self.cap:
                self.cap.release()

            if os.path.exists(self.current_video_file):
                self.cap = cv2.VideoCapture(self.current_video_file)
                self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            else:
                self.cap = None
                fps = 25.0

            self.pipeline = TrafficViolationPipeline(
                fps=fps,
                stop_line=self.stop_line,
                speed_limit_kmh=self.speed_limit,
                meters_per_pixel=self.mpp,
                signal_mode=self.signal_mode,
                initial_state=self.signal_state,
                green_duration=10.0,
                yellow_duration=3.0,
                red_duration=8.0,
                output_violations_dir=VIOLATIONS_DIR,
                detect_interval=2,
                imgsz=480
            )

    def change_video(self, filename: str):
        filepath = os.path.join(VIDEOS_DIR, filename)
        if os.path.exists(filepath):
            clear_violations_dir()  # Automatically clear previous video's snapshots
            self.current_video_file = filepath
            # Adjust default stop line based on video
            if "roads" in filename.lower():
                self.stop_line = ((900, 580), (1350, 580))
            elif "traffic_ip" in filename.lower():
                self.stop_line = ((180, 500), (1120, 500))
            elif "bangalore" in filename.lower():
                self.stop_line = ((150, 480), (1150, 480))
            elif "cuttack" in filename.lower():
                self.stop_line = ((200, 450), (1100, 450))
            elif "hebbal" in filename.lower():
                self.stop_line = ((180, 520), (1180, 520))
            elif "synthetic" in filename.lower():
                self.stop_line = ((280, 520), (1000, 520))
            self.init_pipeline()
            return True
        return False

    def reset(self):
        clear_violations_dir()  # Automatically clear snapshots on reset
        self.init_pipeline()

stream_controller = VideoStreamController()

def frame_generator():
    frame_time = 0.0
    while True:
        if stream_controller.is_paused:
            time.sleep(0.05)
            if stream_controller.last_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + stream_controller.last_frame + b'\r\n')
            continue

        start_t = time.time()
        ret = False
        frame = None

        with stream_controller.lock:
            if stream_controller.cap and stream_controller.cap.isOpened():
                ret, frame = stream_controller.cap.read()
                if not ret:
                    # Loop video
                    stream_controller.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = stream_controller.cap.read()

        if not ret or frame is None:
            time.sleep(0.04)
            continue

        # Process through Deep Learning Pipeline
        with stream_controller.lock:
            if stream_controller.pipeline:
                annotated_frame, stats, _ = stream_controller.pipeline.process_frame(frame)
                stats['video_name'] = os.path.basename(stream_controller.current_video_file)
                stats['frame_w'] = stream_controller.video_width
                stats['frame_h'] = stream_controller.video_height
                stats['fps_actual'] = round(stream_controller.fps_actual, 1)
                stream_controller.current_stats = stats
            else:
                annotated_frame = frame

        # JPEG encode
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        success, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        if success:
            frame_bytes = buffer.tobytes()
            stream_controller.last_frame = frame_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        elapsed = time.time() - start_t
        if elapsed > 0:
            stream_controller.fps_actual = 0.9 * stream_controller.fps_actual + 0.1 * (1.0 / elapsed)

        # Rate control ~ 25 FPS
        time.sleep(max(0.005, 0.035 - elapsed))

@app.route('/')
def index():
    # Automatically clear previous violations and reset stats on browser refresh
    clear_violations_dir()
    stream_controller.reset()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = dict(stream_controller.current_stats)
    return jsonify({
        'status': 'success',
        'stats': stats,
        'is_paused': stream_controller.is_paused,
        'stop_line': stream_controller.stop_line,
        'speed_limit': stream_controller.speed_limit,
        'signal_mode': stream_controller.signal_mode,
        'signal_state': stream_controller.current_stats.get('signal_state', stream_controller.signal_state)
    })

@app.route('/api/violations', methods=['GET'])
def get_violations():
    with stream_controller.lock:
        if stream_controller.pipeline:
            viols = stream_controller.pipeline.violation_detector.violations
        else:
            viols = []
    # Return reverse chronological order
    return jsonify({
        'status': 'success',
        'count': len(viols),
        'violations': list(reversed(viols))
    })

@app.route('/api/set_stop_line', methods=['POST'])
def set_stop_line():
    data = request.json or {}
    x1 = int(data.get('x1', 700))
    y1 = int(data.get('y1', 250))
    x2 = int(data.get('x2', 1200))
    y2 = int(data.get('y2', 250))

    stream_controller.stop_line = ((x1, y1), (x2, y2))
    with stream_controller.lock:
        if stream_controller.pipeline:
            stream_controller.pipeline.set_stop_line((x1, y1), (x2, y2))

    return jsonify({'status': 'success', 'stop_line': stream_controller.stop_line})

@app.route('/api/set_signal', methods=['POST'])
def set_signal():
    data = request.json or {}
    mode = data.get('mode', 'AUTO_CYCLE')
    state = data.get('state', 'GREEN')

    stream_controller.signal_mode = mode
    with stream_controller.lock:
        if stream_controller.pipeline:
            if mode == "MANUAL":
                stream_controller.signal_state = state
                stream_controller.pipeline.set_signal_state(state)
            else:
                stream_controller.pipeline.set_signal_mode("AUTO_CYCLE")

    return jsonify({'status': 'success', 'mode': mode, 'state': state})

@app.route('/api/set_config', methods=['POST'])
def set_config():
    data = request.json or {}
    if 'speed_limit' in data:
        stream_controller.speed_limit = float(data['speed_limit'])
        with stream_controller.lock:
            if stream_controller.pipeline:
                stream_controller.pipeline.set_speed_limit(stream_controller.speed_limit)

    if 'conf_thresh' in data:
        conf = float(data['conf_thresh'])
        with stream_controller.lock:
            if stream_controller.pipeline:
                stream_controller.pipeline.detector.conf_thresh = conf

    if 'meters_per_pixel' in data:
        mpp = float(data['meters_per_pixel'])
        stream_controller.mpp = mpp
        with stream_controller.lock:
            if stream_controller.pipeline:
                stream_controller.pipeline.set_calibration(mpp)

    return jsonify({'status': 'success'})

@app.route('/api/toggle_pause', methods=['POST'])
def toggle_pause():
    stream_controller.is_paused = not stream_controller.is_paused
    return jsonify({'status': 'success', 'is_paused': stream_controller.is_paused})

@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    stream_controller.reset()
    return jsonify({'status': 'success'})

@app.route('/api/clear_violations', methods=['POST'])
def clear_violations_endpoint():
    clear_violations_dir()
    stream_controller.reset()
    return jsonify({'status': 'success', 'message': 'All violation snapshots cleared'})

@app.route('/api/get_videos', methods=['GET'])
def get_videos():
    vids = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    return jsonify({
        'status': 'success',
        'videos': vids,
        'current_video': os.path.basename(stream_controller.current_video_file)
    })

@app.route('/api/select_video', methods=['POST'])
def select_video():
    data = request.json or {}
    filename = data.get('video')
    if filename:
        success = stream_controller.change_video(filename)
        return jsonify({'status': 'success' if success else 'error'})
    return jsonify({'status': 'error', 'message': 'No video specified'}), 400

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

    filename = file.filename
    save_path = os.path.join(VIDEOS_DIR, filename)
    file.save(save_path)
    stream_controller.change_video(filename)
    return jsonify({'status': 'success', 'filename': filename})

@app.route('/violations/<filename>')
def serve_violation_image(filename):
    return send_from_directory(VIOLATIONS_DIR, filename)

@app.route('/api/export_csv', methods=['GET'])
def export_csv():
    with stream_controller.lock:
        if stream_controller.pipeline:
            viols = stream_controller.pipeline.violation_detector.violations
        else:
            viols = []

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow([
        "Violation_ID", "Vehicle_ID", "License_Plate", "Vehicle_Class",
        "Speed_kmh", "Violation_Type", "Fine_Amount_INR", "Signal_State",
        "Frame", "Timestamp", "Evidence_Snapshot"
    ])
    for v in viols:
        cw.writerow([
            v['violation_id'],
            v['track_id'],
            v.get('plate_number', 'N/A'),
            v['class_name'],
            v['speed_kmh'],
            v['violation_type'],
            v.get('fine_amount', 1000),
            v['signal_state'],
            v['frame_idx'],
            v['timestamp'],
            v['snapshot_file']
        ])

    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="traffic_violations_report.csv"
    )

if __name__ == "__main__":
    print("[Flask] Starting Traffic Signal Violation Detection Web Dashboard on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
