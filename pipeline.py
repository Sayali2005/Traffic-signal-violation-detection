"""
Traffic Violation Detection Pipeline.
Coordinates Detection, Tracking, Speed Estimation, Signal Management,
Violation Checking, and Visual Annotation.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from detector import VehicleDetector
from tracker import VehicleTracker, TrackedVehicle
from speed_estimator import SpeedEstimator
from signal_detector import TrafficSignalManager, SignalState
from violation_detector import ViolationDetector
from annotator import VisualAnnotator
from plate_recognizer import LicensePlateRecognizer

class TrafficViolationPipeline:
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        plate_model_name: str = 'models/best_yolov8n.pt',
        conf_thresh: float = 0.35,
        fps: float = 30.0,
        stop_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((200, 680), (1080, 680)),
        speed_limit_kmh: float = 50.0,
        meters_per_pixel: float = 0.045,
        signal_mode: str = "AUTO_CYCLE",
        initial_state: str = SignalState.GREEN,
        green_duration: float = 12.0,
        yellow_duration: float = 3.0,
        red_duration: float = 12.0,
        output_violations_dir: str = "violations",
        detect_interval: int = 2,
        imgsz: int = 480
    ):
        self.detect_interval = max(1, detect_interval)
        self.detector = VehicleDetector(model_name=model_name, conf_thresh=conf_thresh, imgsz=imgsz)
        self.plate_recognizer = LicensePlateRecognizer(model_path=plate_model_name, conf_thresh=0.20, enable_ocr=True)
        self.tracker = VehicleTracker(max_lost=15, iou_thresh=0.3)
        self.speed_estimator = SpeedEstimator(fps=fps, meters_per_pixel=meters_per_pixel, speed_limit_kmh=speed_limit_kmh)
        self.signal_manager = TrafficSignalManager(
            mode=signal_mode,
            initial_state=initial_state,
            green_duration=green_duration,
            yellow_duration=yellow_duration,
            red_duration=red_duration
        )
        self.violation_detector = ViolationDetector(stop_line=stop_line, output_dir=output_violations_dir)
        self.annotator = VisualAnnotator()
        
        self.fps = fps
        self.frame_idx = 0
        self.stop_line = stop_line

    def set_stop_line(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        self.stop_line = (p1, p2)
        self.violation_detector.set_stop_line(p1, p2)

    def set_speed_limit(self, limit: float):
        self.speed_estimator.speed_limit_kmh = limit

    def set_calibration(self, mpp: float):
        self.speed_estimator.meters_per_pixel = mpp

    def set_signal_state(self, state: str):
        self.signal_manager.set_manual_state(state)

    def set_signal_mode(self, mode: str):
        self.signal_manager.set_mode(mode)

    def process_frame(self, frame: np.ndarray, timestamp_sec: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process a single video frame.
        Returns:
            annotated_frame: np.ndarray
            stats: Dict with real-time statistics
            new_violations: List of newly flagged violations in this frame
        """
        self.frame_idx += 1
        elapsed = timestamp_sec if timestamp_sec is not None else (self.frame_idx / self.fps)

        # 1. Update Traffic Signal
        signal_state = self.signal_manager.update(frame=frame, elapsed_seconds=elapsed)

        # 2 & 3. Object Detection & Tracking with Smart Frame Skipping
        is_detect_frame = (self.frame_idx == 1) or (self.frame_idx % self.detect_interval == 0)
        if is_detect_frame:
            raw_detections = self.detector.detect(frame)
            active_vehicles = self.tracker.update(raw_detections, self.frame_idx)
        else:
            active_vehicles = self.tracker.step_skip(self.frame_idx)

        # 4. Estimate Speed & Recognize License Plates
        h, w = frame.shape[:2]
        for v in active_vehicles:
            self.speed_estimator.estimate_speed(v, frame_height=h)

            # Check plate recognition
            vw = v.bbox[2] - v.bbox[0]
            vh = v.bbox[3] - v.bbox[1]
            needs_plate = (
                not getattr(v, 'plate_verified', False) and
                (self.frame_idx - getattr(v, 'plate_last_check_frame', 0) >= 3) and
                (vw >= 40 and vh >= 32)
            )
            is_violator_check = (v.is_violating_signal or v.is_overspeeding) and (getattr(v, 'plate_number', '') in ["SEARCHING...", "DETECTING..."])

            if needs_plate or is_violator_check:
                v.plate_last_check_frame = self.frame_idx
                plate_data = self.plate_recognizer.detect_and_read(frame, v.bbox, track_id=v.track_id)
                v.update_plate(
                    plate_text=plate_data['plate_text'],
                    conf=plate_data['confidence'],
                    plate_bbox=plate_data['plate_bbox'],
                    plate_crop=plate_data['plate_crop'],
                    is_verified=plate_data['is_verified']
                )

        # 5. Check Violations
        new_violations = self.violation_detector.check_violations(
            vehicles=active_vehicles,
            signal_state=signal_state,
            frame=frame,
            frame_idx=self.frame_idx
        )

        # 6. Aggregate Statistics
        sig_viols = sum(1 for item in self.violation_detector.violations if "Signal Jump" in item['violation_type'])
        spd_viols = sum(1 for item in self.violation_detector.violations if "Speed Violation" in item['violation_type'])
        total_fines = sum(item.get('fine_amount', 0) for item in self.violation_detector.violations)
        plates_identified = sum(1 for v in self.tracker.tracks.values() if getattr(v, 'plate_verified', False))
        
        # Vehicle category breakdown
        class_counts = {}
        for v in active_vehicles:
            class_counts[v.class_name] = class_counts.get(v.class_name, 0) + 1

        stats = {
            'frame_idx': self.frame_idx,
            'signal_state': signal_state,
            'signal_mode': self.signal_manager.mode,
            'total_vehicles': self.tracker.total_counted,
            'active_vehicles': len(active_vehicles),
            'total_violations': len(self.violation_detector.violations),
            'signal_violations': sig_viols,
            'speed_violations': spd_viols,
            'total_fines': total_fines,
            'plates_detected': plates_identified,
            'class_breakdown': class_counts,
            'speed_limit': self.speed_estimator.speed_limit_kmh,
            'stop_line': self.stop_line
        }

        # 7. Render Annotations
        annotated_frame = self.annotator.annotate(
            frame=frame,
            vehicles=active_vehicles,
            signal_state=signal_state,
            stop_line=self.stop_line,
            stats=stats
        )

        return annotated_frame, stats, new_violations
