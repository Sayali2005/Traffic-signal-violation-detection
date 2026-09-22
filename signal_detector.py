"""
Traffic Signal Management and Detection Module.
Supports automatic cycling timer, manual interactive control,
and automated HSV-based color detection within a designated traffic light ROI.
"""

from typing import Tuple, Optional
import time
import cv2
import numpy as np

class SignalState:
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class TrafficSignalManager:
    def __init__(
        self,
        mode: str = "AUTO_CYCLE",
        green_duration: float = 12.0,
        yellow_duration: float = 3.0,
        red_duration: float = 12.0,
        initial_state: str = SignalState.GREEN
    ):
        """
        Args:
            mode: "AUTO_CYCLE", "MANUAL", or "COLOR_ROI"
            green_duration: Duration of green light in seconds (for AUTO_CYCLE)
            yellow_duration: Duration of yellow light in seconds
            red_duration: Duration of red light in seconds
            initial_state: Initial signal state
        """
        self.mode = mode
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.red_duration = red_duration
        self.current_state = initial_state
        
        self.state_start_time = time.time()
        self.signal_roi: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

    def set_mode(self, mode: str):
        self.mode = mode

    def set_manual_state(self, state: str):
        if state in [SignalState.RED, SignalState.YELLOW, SignalState.GREEN]:
            self.mode = "MANUAL"
            self.current_state = state
            self.state_start_time = time.time()

    def set_roi(self, x1: int, y1: int, x2: int, y2: int):
        self.signal_roi = (x1, y1, x2, y2)

    def update(self, frame: Optional[np.ndarray] = None, elapsed_seconds: Optional[float] = None) -> str:
        """
        Update and return current signal state.
        If elapsed_seconds is provided (e.g. from video frame timestamp), uses video time instead of wall time.
        """
        if self.mode == "MANUAL":
            return self.current_state

        if self.mode == "COLOR_ROI" and frame is not None and self.signal_roi is not None:
            detected_state = self._detect_light_from_roi(frame, self.signal_roi)
            if detected_state:
                self.current_state = detected_state
            return self.current_state

        # AUTO_CYCLE mode
        now = elapsed_seconds if elapsed_seconds is not None else time.time()
        if elapsed_seconds is not None:
            cycle_time = self.green_duration + self.yellow_duration + self.red_duration
            phase = now % cycle_time
            if phase < self.green_duration:
                self.current_state = SignalState.GREEN
            elif phase < (self.green_duration + self.yellow_duration):
                self.current_state = SignalState.YELLOW
            else:
                self.current_state = SignalState.RED
        else:
            time_in_state = time.time() - self.state_start_time
            if self.current_state == SignalState.GREEN and time_in_state >= self.green_duration:
                self.current_state = SignalState.YELLOW
                self.state_start_time = time.time()
            elif self.current_state == SignalState.YELLOW and time_in_state >= self.yellow_duration:
                self.current_state = SignalState.RED
                self.state_start_time = time.time()
            elif self.current_state == SignalState.RED and time_in_state >= self.red_duration:
                self.current_state = SignalState.GREEN
                self.state_start_time = time.time()

        return self.current_state

    def _detect_light_from_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[str]:
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red spans 0-10 and 160-180 in OpenCV HSV
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)

        # Yellow spans 15-35
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
        yellow_count = cv2.countNonZero(yellow_mask)

        # Green spans 40-85
        green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))
        green_count = cv2.countNonZero(green_mask)

        max_count = max(red_count, yellow_count, green_count)
        if max_count < 10:  # Threshold for noise
            return None

        if max_count == red_count:
            return SignalState.RED
        elif max_count == yellow_count:
            return SignalState.YELLOW
        else:
            return SignalState.GREEN
