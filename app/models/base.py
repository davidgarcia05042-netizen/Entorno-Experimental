"""
Interfaz común para cualquier modelo de estimación de pose.

Al forzar que MediaPipe y YOLOv8 implementen la misma interfaz, el resto
del servicio (video_processor, metrics, la API) no necesita saber cuál
modelo está usando: solo llama a `.predict(frame)` y recibe una lista de
Keypoint ya en el esquema unificado.
"""

from abc import ABC, abstractmethod

import numpy as np

from app.core.keypoint_schema import Keypoint


class PoseEstimator(ABC):
    """Contrato que deben cumplir MediaPipePoseEstimator y YoloV8PoseEstimator."""

    name: str

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Recibe un frame de video (array numpy BGR, como lo entrega OpenCV)
        y devuelve una lista de Keypoint ya normalizados al esquema unificado.

        Si un punto no fue detectado, debe incluirse igual con
        confidence=0.0 y visible=False (no omitirlo), para que las métricas
        de tasa de detección bajo oclusión puedan contarlo correctamente.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Libera recursos del modelo (importante para MediaPipe)."""
        raise NotImplementedError

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
