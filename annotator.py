"""
Visualization and Annotation Module.
Draws green bounding boxes for normal vehicles and red for violating vehicles,
traffic signal indicators, stop lines, speed badges, and HUD telemetry as specified in the paper.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from tracker import TrackedVehicle
from signal_detector import SignalState

class VisualAnnotator:
    # Colors in BGR
    COLOR_NORMAL = (40, 200, 40)       # Green
    COLOR_VIOLATION = (30, 30, 235)    # Red
    COLOR_SPEEDING = (0, 140, 255)     # Amber/Orange
    COLOR_TEXT = (255, 255, 255)
    COLOR_BG_DARK = (20, 24, 30)

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(
        self,
        frame: np.ndarray,
        vehicles: List[TrackedVehicle],
        signal_state: str,
        stop_line: Tuple[Tuple[int, int], Tuple[int, int]],
        stats: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Draw all detection overlays onto the frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Stop Line
        l1, l2 = stop_line
        line_color = (0, 0, 230) if signal_state == SignalState.RED else (40, 200, 40)
        cv2.line(canvas, l1, l2, line_color, 4, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "TRAFFIC STOP LINE",
            (min(l1[0], l2[0]) + 10, min(l1[1], l2[1]) - 10),
            self.font,
            0.55,
            line_color,
            2,
            cv2.LINE_AA
        )

        # 2. Draw Vehicles
        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            is_violator = v.is_violating_signal or v.is_overspeeding

            # Box color styling
            if v.is_violating_signal and v.is_overspeeding:
                box_color = (0, 0, 255)       # Bright Crimson
                badge_bg = (15, 15, 200)
            elif v.is_violating_signal:
                box_color = (30, 30, 245)     # Red
                badge_bg = (20, 20, 180)
            elif v.is_overspeeding:
                box_color = (0, 145, 255)     # Amber/Orange
                badge_bg = (0, 110, 210)
            else:
                box_color = (50, 205, 50)     # Neon Green
                badge_bg = (20, 35, 25)

            # Draw vehicle boundary
            thickness = 3 if is_violator else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            # Draw corner brackets for high-tech surveillance look
            c_len = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
            corner_col = (0, 255, 255) if not is_violator else (255, 255, 255)
            # Top-Left
            cv2.line(canvas, (x1, y1), (x1 + c_len, y1), corner_col, 2)
            cv2.line(canvas, (x1, y1), (x1, y1 + c_len), corner_col, 2)
            # Top-Right
            cv2.line(canvas, (x2, y1), (x2 - c_len, y1), corner_col, 2)
            cv2.line(canvas, (x2, y1), (x2, y1 + c_len), corner_col, 2)
            # Bottom-Left
            cv2.line(canvas, (x1, y2), (x1 + c_len, y2), corner_col, 2)
            cv2.line(canvas, (x1, y2), (x1, y2 - c_len), corner_col, 2)
            # Bottom-Right
            cv2.line(canvas, (x2, y2), (x2 - c_len, y2), corner_col, 2)
            cv2.line(canvas, (x2, y2), (x2, y2 - c_len), corner_col, 2)

            # Draw road contact anchor point
            anchor = v.current_anchor
            cv2.circle(canvas, anchor, 4, (0, 255, 255), -1)

            # Draw License Plate Tag directly on plate if located
            plate_box = getattr(v, 'plate_bbox', None)
            plate_num = getattr(v, 'plate_number', 'SEARCHING...')
            if plate_box is not None:
                px1, py1, px2, py2 = plate_box
                cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(
                    canvas,
                    f"PLATE: {plate_num}",
                    (px1, max(12, py1 - 4)),
                    self.font,
                    0.38,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # Construct Vehicle Identification Badge
            label_main = f"#{v.track_id} [{plate_num}] {v.class_name.upper()} {int(v.speed_kmh)}km/h"
            
            # Offense line if violating
            label_offense = None
            if v.is_violating_signal and v.is_overspeeding:
                label_offense = "[OFFENSE: RED JUMP + SPEED | FINE: INR 2,500]"
            elif v.is_violating_signal:
                label_offense = "[OFFENSE: RED LIGHT JUMP | FINE: INR 1,000]"
            elif v.is_overspeeding:
                label_offense = "[OFFENSE: OVERSPEEDING | FINE: INR 1,500]"

            # Draw Multi-Line Badge
            font_scale = 0.44
            (w_main, h_main), _ = cv2.getTextSize(label_main, self.font, font_scale, 1)
            badge_w = w_main + 14
            badge_h = h_main + 10

            if label_offense:
                (w_off, h_off), _ = cv2.getTextSize(label_offense, self.font, font_scale, 1)
                badge_w = max(badge_w, w_off + 14)
                badge_h += h_off + 6

            badge_y1 = max(0, y1 - badge_h - 4)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w - 2, x1 + badge_w)

            # Badge background with border
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), box_color, 1)

            # Draw text
            text_y1 = badge_y1 + h_main + 4
            cv2.putText(canvas, label_main, (x1 + 6, text_y1), self.font, font_scale, self.COLOR_TEXT, 1, cv2.LINE_AA)

            if label_offense:
                text_y2 = text_y1 + h_off + 6
                cv2.putText(canvas, label_offense, (x1 + 6, text_y2), self.font, font_scale, (0, 220, 255), 1, cv2.LINE_AA)

        # 3. Draw Traffic Signal Indicator (Top-Right)
        self._draw_traffic_light(canvas, signal_state, (w - 120, 20))

        # 4. Draw HUD Telemetry Banner (Top-Left)
        if stats:
            self._draw_hud(canvas, stats, (15, 15))

        return canvas

    def _draw_traffic_light(self, canvas: np.ndarray, signal_state: str, pos: Tuple[int, int]):
        x, y = pos
        # Housing
        cv2.rectangle(canvas, (x, y), (x + 95, y + 150), (25, 25, 25), -1)
        cv2.rectangle(canvas, (x, y), (x + 95, y + 150), (120, 120, 120), 2)

        # Lights: Red, Yellow, Green
        red_color = (0, 0, 255) if signal_state == SignalState.RED else (20, 20, 70)
        cv2.circle(canvas, (x + 47, y + 30), 18, red_color, -1)
        cv2.circle(canvas, (x + 47, y + 30), 18, (180, 180, 180), 1)

        yellow_color = (0, 230, 230) if signal_state == SignalState.YELLOW else (20, 60, 60)
        cv2.circle(canvas, (x + 47, y + 75), 18, yellow_color, -1)
        cv2.circle(canvas, (x + 47, y + 75), 18, (180, 180, 180), 1)

        green_color = (0, 255, 0) if signal_state == SignalState.GREEN else (20, 70, 20)
        cv2.circle(canvas, (x + 47, y + 120), 18, green_color, -1)
        cv2.circle(canvas, (x + 47, y + 120), 18, (180, 180, 180), 1)

    def _draw_hud(self, canvas: np.ndarray, stats: Dict[str, Any], pos: Tuple[int, int]):
        x, y = pos
        box_w, box_h = 340, 155
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (80, 90, 110), 1)

        cv2.putText(canvas, "AI TRAFFIC SURVEILLANCE & ANPR", (x + 12, y + 22), self.font, 0.52, (0, 220, 255), 2, cv2.LINE_AA)

        lines = [
            f"Vehicles Monitored: {stats.get('total_vehicles', 0)}",
            f"Active In Scene: {stats.get('active_vehicles', 0)}",
            f"Plates Identified: {stats.get('plates_detected', 0)}",
            f"Signal Violations: {stats.get('signal_violations', 0)}",
            f"Speed Violations: {stats.get('speed_violations', 0)}",
            f"Total Fines Issued: INR {stats.get('total_fines', 0):,}"
        ]

        for i, line in enumerate(lines):
            line_y = y + 44 + i * 18
            color = (255, 255, 255)
            if "Signal Violations" in line and stats.get('signal_violations', 0) > 0:
                color = (0, 100, 255)
            elif "Speed Violations" in line and stats.get('speed_violations', 0) > 0:
                color = (0, 180, 255)
            elif "Total Fines" in line:
                color = (0, 230, 255) if stats.get('total_fines', 0) > 0 else (200, 200, 200)
            elif "Plates Identified" in line:
                color = (120, 240, 120)
            cv2.putText(canvas, line, (x + 12, line_y), self.font, 0.42, color, 1, cv2.LINE_AA)
