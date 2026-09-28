"""
Traffic Violation Detection Module.
Detects Signal Jump (crossing stop line during Red light) and Speeding violations.
Crops evidence snapshots and logs violations.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import cv2
import numpy as np
from tracker import TrackedVehicle
from signal_detector import SignalState

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Return True if line segments AB and CD intersect."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def point_line_side(p: Tuple[float, float], l1: Tuple[float, float], l2: Tuple[float, float]) -> float:
    """Signed cross product determining which side of line (l1->l2) point p lies on."""
    return (l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0])

def get_violation_fine(violation_label: str) -> int:
    """Calculate fine amount based on offense severity."""
    if "Signal Jump" in violation_label and "Speed Violation" in violation_label:
        return 2500
    elif "Signal Jump" in violation_label:
        return 1000
    elif "Speed Violation" in violation_label:
        return 1500
    return 1000

class ViolationDetector:
    def __init__(
        self,
        stop_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((200, 700), (1100, 700)),
        output_dir: str = "violations"
    ):
        """
        Args:
            stop_line: ((x1, y1), (x2, y2)) coordinates of the traffic stop line
            output_dir: Directory to save violation evidence snapshots
        """
        self.stop_line = stop_line
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.violations: List[Dict[str, Any]] = []
        self.violation_counter = 0

    def set_stop_line(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        self.stop_line = (p1, p2)

    def check_violations(
        self,
        vehicles: List[TrackedVehicle],
        signal_state: str,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all active vehicles for traffic violations.
        """
        new_violations = []
        current_time = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        l1, l2 = self.stop_line

        for v in vehicles:
            # 1. Signal Jump Violation Check
            if not v.has_crossed_stop_line and len(v.history) >= 2:
                curr_anchor = v.current_anchor
                
                # Check intersection with path from multiple previous history steps
                intersected = False
                lookback = min(len(v.history), 8)
                for step in range(1, lookback):
                    prev_anchor = v.history[-1 - step][1]
                    if segments_intersect(prev_anchor, curr_anchor, l1, l2):
                        intersected = True
                        break
                    
                    # Also check signed distance crossing within segment extent
                    side_prev = point_line_side(prev_anchor, l1, l2)
                    side_curr = point_line_side(curr_anchor, l1, l2)
                    if (side_prev * side_curr) <= 0 and side_prev != side_curr:
                        # Crossed the infinite line; verify within segment x/y bounds with margin
                        min_x, max_x = min(l1[0], l2[0]) - 50, max(l1[0], l2[0]) + 50
                        min_y, max_y = min(l1[1], l2[1]) - 50, max(l1[1], l2[1]) + 50
                        if (min_x <= curr_anchor[0] <= max_x) and (min_y <= curr_anchor[1] <= max_y):
                            intersected = True
                            break

                # Also check if bounding box bottom intersects the line
                if not intersected:
                    bx1, by1, bx2, by2 = v.bbox
                    if segments_intersect((bx1, by2), (bx2, by2), l1, l2):
                        intersected = True

                if intersected:
                    v.has_crossed_stop_line = True
                    v.crossing_frame = frame_idx

                    # If light is RED at the moment of crossing, it's a SIGNAL JUMP!
                    if signal_state == SignalState.RED:
                        v.is_violating_signal = True
                        if "Signal Jump" not in v.violation_types:
                            v.violation_types.append("Signal Jump")

            # 2. Check Signal Jump violation logging (independent of speed violation)
            if v.is_violating_signal and not getattr(v, 'signal_violation_logged', False):
                v.signal_violation_logged = True
                v.snapshot_saved = True
                self.violation_counter += 1

                violation_label = "Signal Jump & Speed Violation" if v.is_overspeeding else "Signal Jump"
                fine_val = get_violation_fine(violation_label)
                plate_txt = getattr(v, 'plate_number', 'DETECTING...')
                snapshot_filename = f"viol_{self.violation_counter}_id{v.track_id}_{plate_txt}_{v.class_name}.jpg"
                snapshot_path = os.path.join(self.output_dir, snapshot_filename)

                # Save cropped evidence
                self._save_evidence_snapshot(frame, v, snapshot_path, violation_label, fine_val, current_time, signal_state)

                record = {
                    'violation_id': self.violation_counter,
                    'track_id': v.track_id,
                    'plate_number': plate_txt,
                    'plate_verified': getattr(v, 'plate_verified', False),
                    'class_name': v.class_name,
                    'speed_kmh': round(v.speed_kmh, 1),
                    'violation_type': violation_label,
                    'fine_amount': fine_val,
                    'fine_str': f"₹{fine_val:,}",
                    'signal_state': signal_state,
                    'frame_idx': frame_idx,
                    'timestamp': current_time,
                    'bbox': v.bbox,
                    'snapshot_file': snapshot_filename
                }
                self.violations.append(record)
                new_violations.append(record)

            # 3. Check Speed violation logging (independent of signal violation)
            elif v.is_overspeeding and not getattr(v, 'speed_violation_logged', False):
                v.speed_violation_logged = True
                v.snapshot_saved = True
                self.violation_counter += 1

                violation_label = "Speed Violation"
                fine_val = get_violation_fine(violation_label)
                plate_txt = getattr(v, 'plate_number', 'DETECTING...')
                snapshot_filename = f"speeding_{self.violation_counter}_id{v.track_id}_{plate_txt}_{v.class_name}.jpg"
                snapshot_path = os.path.join(self.output_dir, snapshot_filename)

                # Save cropped evidence
                self._save_evidence_snapshot(frame, v, snapshot_path, violation_label, fine_val, current_time, signal_state)

                record = {
                    'violation_id': self.violation_counter,
                    'track_id': v.track_id,
                    'plate_number': plate_txt,
                    'plate_verified': getattr(v, 'plate_verified', False),
                    'class_name': v.class_name,
                    'speed_kmh': round(v.speed_kmh, 1),
                    'violation_type': violation_label,
                    'fine_amount': fine_val,
                    'fine_str': f"₹{fine_val:,}",
                    'signal_state': signal_state,
                    'frame_idx': frame_idx,
                    'timestamp': current_time,
                    'bbox': v.bbox,
                    'snapshot_file': snapshot_filename
                }
                self.violations.append(record)
                new_violations.append(record)

        return new_violations

    def _save_evidence_snapshot(
        self,
        frame: np.ndarray,
        vehicle: TrackedVehicle,
        output_path: str,
        violation_label: str,
        fine_amount: int,
        timestamp: str,
        signal_state: str = "RED"
    ):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = vehicle.bbox

        # Add margin around vehicle
        pad_x = int((x2 - x1) * 0.35)
        pad_y = int((y2 - y1) * 0.35)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            return

        # Draw red border on evidence crop
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (0, 0, 235), 3)

        # Highlight vehicle bounding box in crop
        vx1_rel = max(2, x1 - crop_x1)
        vy1_rel = max(2, y1 - crop_y1)
        vx2_rel = min(crop.shape[1] - 3, x2 - crop_x1)
        vy2_rel = min(crop.shape[0] - 3, y2 - crop_y1)
        cv2.rectangle(crop, (vx1_rel, vy1_rel), (vx2_rel, vy2_rel), (0, 70, 255), 2)

        # Ensure minimum width of 500px for crystal-clear header text readability
        if crop.shape[1] < 500:
            diff = 500 - crop.shape[1]
            pad_l = diff // 2
            pad_r = diff - pad_l
            crop = cv2.copyMakeBorder(crop, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(14, 16, 22))

        # If vehicle has a recognized plate crop, prepare an inset badge
        plate_crop = getattr(vehicle, 'plate_crop', None)
        if plate_crop is not None and plate_crop.size > 0:
            try:
                # Resize plate crop to standard height of 45px
                target_h = 45
                p_aspect = plate_crop.shape[1] / max(1, plate_crop.shape[0])
                target_w = max(90, min(180, int(target_h * p_aspect)))
                p_resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                
                # Place at bottom-right of the crop if space permits
                if crop.shape[0] > target_h + 10 and crop.shape[1] > target_w + 10:
                    px_start = crop.shape[1] - target_w - 10
                    py_start = crop.shape[0] - target_h - 10
                    # Border around plate inset
                    cv2.rectangle(crop, (px_start - 2, py_start - 2), (px_start + target_w + 2, py_start + target_h + 2), (0, 255, 255), 2)
                    crop[py_start:py_start + target_h, px_start:px_start + target_w] = p_resized
                    cv2.putText(crop, "PLATE", (px_start, py_start - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

        # High-definition header banner
        banner_h = 62
        banner = np.zeros((banner_h, crop.shape[1], 3), dtype=np.uint8)
        banner[:] = (18, 22, 28)

        font = cv2.FONT_HERSHEY_SIMPLEX
        plate_str = getattr(vehicle, 'plate_number', 'N/A')
        text_top = f"VEHICLE #{vehicle.track_id} {vehicle.class_name.upper()} | PLATE: {plate_str}"
        text_mid = f"OFFENSE: {violation_label} | FINE CHARGE: INR {fine_amount:,}"
        text_bot = f"SPEED: {vehicle.speed_kmh} km/h | SIGNAL: {signal_state} | TIME: {timestamp}"

        cv2.putText(banner, text_top, (10, 18), font, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, text_mid, (10, 38), font, 0.46, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, text_bot, (10, 56), font, 0.38, (180, 190, 205), 1, cv2.LINE_AA)

        combined = np.vstack([banner, crop])
        cv2.imwrite(output_path, combined)
