"""
Wrapper de MediaPipe Pose.

Requiere: pip install mediapipe opencv-python-headless numpy
"""

import numpy as np

from app.core.keypoint_schema import MEDIAPIPE_INDEX, COCO_ORDER, Keypoint
from app.models.base import PoseEstimator


class MediaPipePoseEstimator(PoseEstimator):
    name = "mediapipe_pose"

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        # Import perezoso: así el resto del servicio puede importarse sin
        # tener mediapipe instalado (útil para tests que no lo requieren).
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        height, width = frame.shape[:2]

        # MediaPipe espera RGB, OpenCV entrega BGR
        rgb_frame = frame[:, :, ::-1]
        results = self._pose.process(rgb_frame)

        keypoints: list[Keypoint] = []

        if results.pose_landmarks is None:
            # El modelo no detectó a la persona en absoluto: se devuelven
            # los 17 puntos marcados como no visibles, no se omiten.
            for name in COCO_ORDER:
                keypoints.append(Keypoint(name=name, x=0.0, y=0.0, confidence=0.0, visible=False))
            return keypoints

        landmarks = results.pose_landmarks.landmark

        for name in COCO_ORDER:
            idx = MEDIAPIPE_INDEX[name]
            lm = landmarks[idx]
            keypoints.append(
                Keypoint(
                    name=name,
                    x=lm.x * width,
                    y=lm.y * height,
                    confidence=float(lm.visibility),  # MediaPipe usa "visibility" como proxy de confianza
                    visible=lm.visibility >= 0.5,
                )
            )

        return keypoints

    def close(self) -> None:
        self._pose.close()
