"""
Wrapper de YOLOv8-Pose (librería Ultralytics).

Requiere: pip install ultralytics opencv-python-headless numpy

Nota sobre pesos del modelo: Ultralytics descarga automáticamente
'yolov8n-pose.pt' (variante nano, más rápida) la primera vez que se usa.
Para el comparativo también pueden evaluar 'yolov8s-pose.pt' o
'yolov8m-pose.pt' si quieren ver el trade-off precisión/latencia entre
tamaños del mismo modelo, no solo entre MediaPipe y YOLOv8.

Esta misma clase también sirve para YOLO11-Pose (pasar
weights='yolo11n-pose.pt', etc.) -- Ultralytics expone ambas familias
bajo la misma clase `YOLO`, y YOLO11-Pose produce el mismo esquema
COCO-17 que YOLOv8-Pose, así que no hace falta un wrapper aparte.
Requiere ultralytics>=8.3.0 (no incluía las configs de YOLO11 antes).
"""

import numpy as np

from app.core.keypoint_schema import COCO_ORDER, Keypoint
from app.models.base import PoseEstimator


class YoloV8PoseEstimator(PoseEstimator):
    name = "yolov8_pose"

    def __init__(
        self,
        weights: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.5,
        inference_conf: float | None = None,
        inference_iou: float | None = None,
    ):
        """
        `inference_conf`/`inference_iou`: umbrales pasados directamente a
        `model.predict()` (filtran detecciones de PERSONA antes de que
        exista un keypoint que evaluar). Si se dejan en None, se usan los
        defaults internos de Ultralytics (igual que el comportamiento
        histórico de este wrapper). Distinto de `confidence_threshold`,
        que solo marca `visible` en un keypoint YA calculado -- no evita
        que una detección de persona débil/ruidosa entre al cálculo.
        """
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._confidence_threshold = confidence_threshold
        self._inference_conf = inference_conf
        self._inference_iou = inference_iou

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        predict_kwargs = {"verbose": False}
        if self._inference_conf is not None:
            predict_kwargs["conf"] = self._inference_conf
        if self._inference_iou is not None:
            predict_kwargs["iou"] = self._inference_iou
        results = self._model(frame, **predict_kwargs)[0]

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
