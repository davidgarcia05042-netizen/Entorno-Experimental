"""
Wrapper de YOLOv8-Pose (librería Ultralytics).

Requiere: pip install ultralytics opencv-python-headless numpy

Nota sobre pesos del modelo: Ultralytics descarga automáticamente
'yolov8n-pose.pt' (variante nano, más rápida) la primera vez que se usa.
Para el comparativo también pueden evaluar 'yolov8s-pose.pt' o
'yolov8m-pose.pt' si quieren ver el trade-off precisión/latencia entre
tamaños del mismo modelo, no solo entre MediaPipe y YOLOv8.
"""

import numpy as np

from app.core.keypoint_schema import COCO_ORDER, Keypoint
from app.models.base import PoseEstimator


class YoloV8PoseEstimator(PoseEstimator):
    name = "yolov8_pose"

    def __init__(self, weights: str = "yolov8n-pose.pt", confidence_threshold: float = 0.5):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._confidence_threshold = confidence_threshold

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        results = self._model(frame, verbose=False)[0]

        keypoints: list[Keypoint] = []

        if results.keypoints is None or len(results.keypoints.data) == 0:
            for name in COCO_ORDER:
                keypoints.append(Keypoint(name=name, x=0.0, y=0.0, confidence=0.0, visible=False))
            return keypoints

        # Si hay varias personas detectadas, se toma la de mayor confianza
        # de bounding box (asumiendo un solo paciente en cuadro, como es
        # el caso esperado en telefisioterapia).
        person_idx = 0
        if results.boxes is not None and len(results.boxes.conf) > 1:
            person_idx = int(results.boxes.conf.argmax())

        kpts_xy = results.keypoints.xy[person_idx].cpu().numpy()  # (17, 2)
        kpts_conf = results.keypoints.conf[person_idx].cpu().numpy()  # (17,)

        for i, name in enumerate(COCO_ORDER):
            x, y = kpts_xy[i]
            conf = float(kpts_conf[i])
            keypoints.append(
                Keypoint(
                    name=name,
                    x=float(x),
                    y=float(y),
                    confidence=conf,
                    visible=conf >= self._confidence_threshold,
                )
            )

        return keypoints

    def close(self) -> None:
        # Ultralytics no requiere liberar recursos explícitamente
        pass
