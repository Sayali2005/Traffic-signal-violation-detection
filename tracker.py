"""
Multi-Object Tracker for Traffic Participants.
Tracks vehicle bounding boxes, centroids, and road contact anchor points across consecutive frames.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def calculate_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class TrackedVehicle:
    def __init__(self, track_id: int, bbox: List[int], class_id: int, class_name: str, conf: float, frame_idx: int):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.conf = conf
        self.bbox = bbox  # [x1, y1, x2, y2]
        
        # History: list of (frame_idx, anchor_point (x, y), bbox)
        # Anchor point is the bottom-center of the bounding box on the road plane
        anchor = ((bbox[0] + bbox[2]) // 2, bbox[3])
        self.history: List[Tuple[int, Tuple[int, int], List[int]]] = [(frame_idx, anchor, bbox)]
        
        self.lost_frames = 0
        self.speed_kmh: float = 0.0
        self.is_violating_signal = False
        self.is_overspeeding = False
        self.has_crossed_stop_line = False
        self.crossing_frame: Optional[int] = None
        self.violation_types: List[str] = []
        self.snapshot_saved = False
        self.signal_violation_logged = False
        self.speed_violation_logged = False
        self.overspeed_count = 0
        self.is_confirmed = False

    @property
    def current_anchor(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, self.bbox[3])

    @property
    def current_centroid(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    def update(self, bbox: List[int], conf: float, frame_idx: int):
        self.bbox = bbox
        self.conf = conf
        self.lost_frames = 0
        anchor = ((bbox[0] + bbox[2]) // 2, bbox[3])
        self.history.append((frame_idx, anchor, bbox))
        if len(self.history) >= 3:
            self.is_confirmed = True
        # Keep last 100 history points
        if len(self.history) > 100:
            self.history.pop(0)

    def extrapolate(self, frame_idx: int):
        """
        Extrapolate vehicle position on skipped detection frames using recent velocity.
        """
        if len(self.history) >= 2:
            prev_f, prev_anchor, prev_box = self.history[-2]
            curr_f, curr_anchor, curr_box = self.history[-1]
            df = max(1, curr_f - prev_f)
            dx = (curr_anchor[0] - prev_anchor[0]) / df
            dy = (curr_anchor[1] - prev_anchor[1]) / df
            dt = max(1, frame_idx - curr_f)
            
            # Predict new bounding box
            shift_x = int(round(dx * dt))
            shift_y = int(round(dy * dt))
            new_bbox = [
                self.bbox[0] + shift_x,
                self.bbox[1] + shift_y,
                self.bbox[2] + shift_x,
                self.bbox[3] + shift_y
            ]
            self.bbox = new_bbox
            new_anchor = ((new_bbox[0] + new_bbox[2]) // 2, new_bbox[3])
            self.history.append((frame_idx, new_anchor, new_bbox))
            if len(self.history) > 100:
                self.history.pop(0)
        else:
            self.history.append((frame_idx, self.current_anchor, self.bbox))
            if len(self.history) > 100:
                self.history.pop(0)

class VehicleTracker:
    def __init__(self, max_lost: int = 30, iou_thresh: float = 0.2):
        self.next_id = 1
        self.tracks: Dict[int, TrackedVehicle] = {}
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.total_counted = 0
        self.counted_ids = set()

    def step_skip(self, frame_idx: int) -> List[TrackedVehicle]:
        """
        Update active tracks on frames where detection is skipped.
        Extrapolates positions without accumulating lost_frames.
        """
        for track in list(self.tracks.values()):
            track.extrapolate(frame_idx)
        return list(self.tracks.values())

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[TrackedVehicle]:
        """
        Match incoming detections with existing tracks using a two-stage matching:
        Stage 1: IoU bounding box overlap matching.
        Stage 2: Spatial centroid/anchor Euclidean distance fallback.
        """
        # If no tracks exist yet, initialize all detections as new tracks
        if len(self.tracks) == 0:
            for det in detections:
                track = TrackedVehicle(
                    track_id=self.next_id,
                    bbox=det['bbox'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    conf=det['conf'],
                    frame_idx=frame_idx
                )
                self.tracks[self.next_id] = track
                self.next_id += 1

            self._update_confirmed_counts()
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))

        if len(detections) == 0:
            for t_id in track_ids:
                self.tracks[t_id].lost_frames += 1
                if self.tracks[t_id].lost_frames > self.max_lost:
                    del self.tracks[t_id]
            return list(self.tracks.values())

        # Stage 1: Greedy IoU matching
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, t_id in enumerate(track_ids):
            track = self.tracks[t_id]
            for j, det in enumerate(detections):
                iou = calculate_iou(track.bbox, det['bbox'])
                cost_matrix[i, j] = iou

        matched_tracks = set()
        matched_dets = set()

        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if cost_matrix[i, j] >= self.iou_thresh:
                    matches.append((cost_matrix[i, j], i, j))
        matches.sort(key=lambda x: x[0], reverse=True)

        for iou, t_idx, d_idx in matches:
            t_id = track_ids[t_idx]
            if t_id in matched_tracks or d_idx in matched_dets:
                continue
            det = detections[d_idx]
            self.tracks[t_id].update(det['bbox'], det['conf'], frame_idx)
            matched_tracks.add(t_id)
            matched_dets.add(d_idx)

        # Stage 2: Spatial Euclidean distance fallback for unmatched detections
        unmatched_track_ids = [t_id for t_id in track_ids if t_id not in matched_tracks]
        unmatched_det_indices = [d_idx for d_idx in det_indices if d_idx not in matched_dets]

        for d_idx in unmatched_det_indices:
            det = detections[d_idx]
            det_anchor = ((det['bbox'][0] + det['bbox'][2]) // 2, det['bbox'][3])
            best_dist = float('inf')
            best_t_id = None

            for t_id in unmatched_track_ids:
                track = self.tracks[t_id]
                # Match same class or generic vehicle category
                if track.class_name != det['class_name'] and not (
                    track.class_name in ['car', 'bus', 'truck'] and det['class_name'] in ['car', 'bus', 'truck']
                ):
                    continue

                t_anchor = track.current_anchor
                dist = np.hypot(det_anchor[0] - t_anchor[0], det_anchor[1] - t_anchor[1])
                # Allow distance up to 1.5x bbox diagonal or 100 pixels
                bbox_diag = np.hypot(det['bbox'][2] - det['bbox'][0], det['bbox'][3] - det['bbox'][1])
                max_allowed_dist = max(80.0, bbox_diag * 1.5)

                if dist < max_allowed_dist and dist < best_dist:
                    best_dist = dist
                    best_t_id = t_id

            if best_t_id is not None:
                self.tracks[best_t_id].update(det['bbox'], det['conf'], frame_idx)
                matched_tracks.add(best_t_id)
                matched_dets.add(d_idx)
                unmatched_track_ids.remove(best_t_id)

        # Handle unmatched existing tracks
        for t_id in track_ids:
            if t_id not in matched_tracks:
                self.tracks[t_id].lost_frames += 1
                if self.tracks[t_id].lost_frames > self.max_lost:
                    del self.tracks[t_id]

        # Handle unmatched detections (create new tracks)
        for d_idx in det_indices:
            if d_idx not in matched_dets:
                det = detections[d_idx]
                track = TrackedVehicle(
                    track_id=self.next_id,
                    bbox=det['bbox'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    conf=det['conf'],
                    frame_idx=frame_idx
                )
                self.tracks[self.next_id] = track
                self.next_id += 1

        self._update_confirmed_counts()
        return list(self.tracks.values())

    def _update_confirmed_counts(self):
        """Count unique vehicles once confirmed (sustained detection across frames)."""
        for t_id, track in self.tracks.items():
            if len(track.history) >= 2 and t_id not in self.counted_ids:
                self.counted_ids.add(t_id)
                self.total_counted += 1
