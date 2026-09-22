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
                snapshot_filename = f"signal_jump_{self.violation_counter}_id{v.track_id}_{v.class_name}.jpg"
                snapshot_path = os.path.join(self.output_dir, snapshot_filename)

                # Save cropped evidence
                self._save_evidence_snapshot(frame, v, snapshot_path, violation_label, current_time)

                record = {
                    'violation_id': self.violation_counter,
                    'track_id': v.track_id,
                    'class_name': v.class_name,
                    'speed_kmh': v.speed_kmh,
                    'violation_type': violation_label,
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
                snapshot_filename = f"speeding_{self.violation_counter}_id{v.track_id}_{v.class_name}.jpg"
                snapshot_path = os.path.join(self.output_dir, snapshot_filename)

                # Save cropped evidence
                self._save_evidence_snapshot(frame, v, snapshot_path, violation_label, current_time)

                record = {
                    'violation_id': self.violation_counter,
                    'track_id': v.track_id,
                    'class_name': v.class_name,
                    'speed_kmh': v.speed_kmh,
                    'violation_type': violation_label,
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
        timestamp: str
    ):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = vehicle.bbox

        # Add margin around vehicle
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.3)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            return

        # Draw red border on evidence crop
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (0, 0, 255), 3)

        # Header banner
        banner_h = 42
        banner = np.zeros((banner_h, crop.shape[1], 3), dtype=np.uint8)
        banner[:] = (20, 20, 30)
        
        # Overlay text on banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_top = f"ID #{vehicle.track_id} {vehicle.class_name.upper()} | {vehicle.speed_kmh} km/h"
        text_sub = f"VIOLATION: {violation_label}"
        cv2.putText(banner, text_top, (8, 16), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, text_sub, (8, 34), font, 0.45, (0, 70, 255), 1, cv2.LINE_AA)

        combined = np.vstack([banner, crop])
        cv2.imwrite(output_path, combined)
