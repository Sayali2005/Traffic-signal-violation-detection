"""
Speed Estimation Module.
Estimates vehicle velocity using spatial anchor-point displacement,
temporal frame intervals (FPS), and perspective distance calibration.
"""

from typing import List, Tuple, Optional
import math
from tracker import TrackedVehicle

class SpeedEstimator:
    def __init__(self, fps: float = 30.0, meters_per_pixel: float = 0.05, speed_limit_kmh: float = 50.0, window_frames: int = 10):
        """
        Args:
            fps: Video frames per second
            meters_per_pixel: Base calibration scale (meters per pixel on road plane)
            speed_limit_kmh: Speed violation threshold in km/h
            window_frames: Sliding window of frames used to calculate stable velocity
        """
        self.fps = fps if fps > 0 else 30.0
        self.meters_per_pixel = meters_per_pixel
        self.speed_limit_kmh = speed_limit_kmh
        self.window_frames = window_frames

    def estimate_speed(self, vehicle: TrackedVehicle, frame_height: int = 1080) -> float:
        """
        Calculate vehicle speed in km/h based on recent history.
        Applies a perspective weighting factor based on vertical position (y-coordinate)
        because objects near the camera (bottom of frame) traverse more pixels per meter
        than objects in the distance (top of frame).
        """
        if len(vehicle.history) < 3:
            return vehicle.speed_kmh

        # Select window of history
        sample_len = min(len(vehicle.history), self.window_frames)
        prev_entry = vehicle.history[-sample_len]
        curr_entry = vehicle.history[-1]

        prev_frame, prev_pos, prev_bbox = prev_entry
        curr_frame, curr_pos, curr_bbox = curr_entry

        delta_frames = curr_frame - prev_frame
        if delta_frames <= 0:
            return vehicle.speed_kmh

        time_seconds = delta_frames / self.fps

        # Anchor displacement
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        pixel_distance = math.sqrt(dx * dx + dy * dy)

        # Filter out minor detection jitter for stationary / parked vehicles
        if pixel_distance < 6.0:
            speed_kmh = 0.0
        else:
            # Perspective correction factor:
            # Objects at bottom (y ~ frame_height) have smaller meters_per_pixel (more pixels per meter).
            # Objects near top (y ~ frame_height * 0.2) have larger meters_per_pixel (fewer pixels per meter).
            avg_y = (curr_pos[1] + prev_pos[1]) / 2.0
            y_ratio = max(0.1, avg_y / max(1.0, float(frame_height)))
            # Perspective scale: at bottom (y_ratio ~ 1.0) scale is normal; at top (y_ratio ~ 0.3) scale is higher
            perspective_factor = 1.0 / (0.4 + 0.6 * y_ratio)

            distance_meters = pixel_distance * self.meters_per_pixel * perspective_factor
            speed_mps = distance_meters / time_seconds
            speed_kmh = speed_mps * 3.6

            # Physical sanity check: ignore sudden bounding-box jumps when entering camera borders
            if speed_kmh > 150.0:
                speed_kmh = min(vehicle.speed_kmh, 120.0) if vehicle.speed_kmh > 0 else 45.0

        # Smooth updating with exponential moving average
        if vehicle.speed_kmh > 0:
            smoothed_speed = 0.7 * vehicle.speed_kmh + 0.3 * speed_kmh
        else:
            smoothed_speed = speed_kmh

        vehicle.speed_kmh = round(smoothed_speed, 1)

        # Check speed violation with debouncing (must overspeed for at least 2 consecutive checks)
        if vehicle.speed_kmh > self.speed_limit_kmh:
            vehicle.overspeed_count = getattr(vehicle, 'overspeed_count', 0) + 1
            if vehicle.overspeed_count >= 2:
                vehicle.is_overspeeding = True
                if "Speed Violation" not in vehicle.violation_types:
                    vehicle.violation_types.append("Speed Violation")
        else:
            vehicle.overspeed_count = max(0, getattr(vehicle, 'overspeed_count', 0) - 1)

        return vehicle.speed_kmh
