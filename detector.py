"""
Vehicle Detection Module using Ultralytics YOLO.
Detects traffic participants: cars, motorcycles, buses, trucks, and bicycles.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from ultralytics import YOLO

class VehicleDetector:
    # COCO Class IDs for vehicles
    # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    DEFAULT_VEHICLE_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, model_name: str = 'yolov8n.pt', conf_thresh: float = 0.35, device: str = None, imgsz: int = 480):
        """
        Initialize the YOLO detector.
        Args:
            model_name: Path or model tag (e.g. 'yolov8n.pt', 'yolov8s.pt')
            conf_thresh: Confidence score threshold
            device: 'cpu', 'cuda', or None for auto-detection
            imgsz: Input inference image resolution (e.g. 480 for 2-3x faster CPU inference)
        """
        self.conf_thresh = conf_thresh
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.imgsz = imgsz
        print(f"[VehicleDetector] Loading model '{model_name}' on {self.device} (imgsz={imgsz})...")
        self.model = YOLO(model_name)
        self.target_classes = set(self.DEFAULT_VEHICLE_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in a single frame.
        Returns a list of dicts:
            {
                'bbox': [x1, y1, x2, y2],
                'conf': float,
                'class_id': int,
                'class_name': str
            }
        """
        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                classes=list(self.target_classes),
                device=self.device,
                imgsz=self.imgsz,
                verbose=False
            )

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': round(conf, 3),
                    'class_id': cls_id,
                    'class_name': self.DEFAULT_VEHICLE_CLASSES.get(cls_id, 'vehicle')
                })

        return detections
