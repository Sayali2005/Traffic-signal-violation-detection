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

            # As per paper: Green normally, Red on violation
            box_color = self.COLOR_VIOLATION if is_violator else self.COLOR_NORMAL
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2 if not is_violator else 3)

            # Draw Anchor Point
            anchor = v.current_anchor
            cv2.circle(canvas, anchor, 4, (0, 255, 255), -1)

            # Build label
            label = f"#{v.track_id} {v.class_name} {int(v.speed_kmh)}km/h"
            if v.is_violating_signal and v.is_overspeeding:
                label += " [RED JUMP + SPEED]"
            elif v.is_violating_signal:
                label += " [RED LIGHT VIOLATION]"
            elif v.is_overspeeding:
                label += " [SPEED VIOLATION]"

            # Label badge
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.45, 1)
            cv2.rectangle(
                canvas,
                (x1, max(0, y1 - th - 8)),
                (x1 + tw + 10, y1),
                box_color,
                -1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 5, max(12, y1 - 4)),
                self.font,
                0.45,
                self.COLOR_TEXT,
                1,
                cv2.LINE_AA
            )

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
        # Red
        red_color = (0, 0, 255) if signal_state == SignalState.RED else (20, 20, 70)
        cv2.circle(canvas, (x + 47, y + 30), 18, red_color, -1)
        cv2.circle(canvas, (x + 47, y + 30), 18, (180, 180, 180), 1)

        # Yellow
        yellow_color = (0, 230, 230) if signal_state == SignalState.YELLOW else (20, 60, 60)
        cv2.circle(canvas, (x + 47, y + 75), 18, yellow_color, -1)
        cv2.circle(canvas, (x + 47, y + 75), 18, (180, 180, 180), 1)

        # Green
        green_color = (0, 255, 0) if signal_state == SignalState.GREEN else (20, 70, 20)
        cv2.circle(canvas, (x + 47, y + 120), 18, green_color, -1)
        cv2.circle(canvas, (x + 47, y + 120), 18, (180, 180, 180), 1)

    def _draw_hud(self, canvas: np.ndarray, stats: Dict[str, Any], pos: Tuple[int, int]):
        x, y = pos
        box_w, box_h = 320, 125
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (80, 90, 110), 1)

        cv2.putText(canvas, "TRAFFIC VIOLATION SYSTEM", (x + 12, y + 22), self.font, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        lines = [
            f"Total Vehicles Counted: {stats.get('total_vehicles', 0)}",
            f"Active In Scene: {stats.get('active_vehicles', 0)}",
            f"Signal Violations: {stats.get('signal_violations', 0)}",
            f"Speed Violations: {stats.get('speed_violations', 0)}"
        ]

        for i, line in enumerate(lines):
            line_y = y + 46 + i * 18
            color = (255, 255, 255)
            if "Signal Violations" in line and stats.get('signal_violations', 0) > 0:
                color = (0, 100, 255)
            elif "Speed Violations" in line and stats.get('speed_violations', 0) > 0:
                color = (0, 180, 255)
            cv2.putText(canvas, line, (x + 12, line_y), self.font, 0.44, color, 1, cv2.LINE_AA)
