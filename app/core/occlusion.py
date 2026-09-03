"""
Simulación de oclusión parcial programada.

Este módulo cubre el caso "oclusión programada" de su diseño mixto
(oclusión natural ya presente en las tomas + oclusión generada por código).
La oclusión natural NO se simula aquí: simplemente se documenta como
metadata del video (campo `occlusion_applied="natural"` en FrameResult)
cuando ya viene en el material original del laboratorio.

Dos formas de ocluir soportadas:
  1. Por región fija del frame (ej. "tercio inferior"), útil cuando aún
     no se tienen keypoints de referencia.
  2. Por articulación objetivo + radio, útil cuando ya se corrió una
     primera pasada de detección y se quiere ocluir deliberadamente una
     zona anatómica específica (ej. "ocluir la rodilla izquierda").
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from app.core.keypoint_schema import UnifiedKeypoint
from app.models.mediapipe_pose import MediaPipePoseEstimator


class OcclusionMethod(str, Enum):
    BLACK_BOX = "black_box"       # rectángulo sólido negro
    GAUSSIAN_BLUR = "gaussian_blur"  # difuminado fuerte (oclusión "suave", ej. por otro objeto translúcido)


@dataclass
class OcclusionRegion:
    """Región del frame a ocluir, en coordenadas de píxel."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


def region_from_fixed_fraction(
    frame_shape: tuple[int, int],
    x_fraction_range: tuple[float, float],
    y_fraction_range: tuple[float, float],
) -> OcclusionRegion:
    """
    Define una región a partir de fracciones del frame (0.0 a 1.0).
    Ej: ocluir el tercio inferior -> y_fraction_range=(0.66, 1.0), x_fraction_range=(0.0, 1.0)
    """
    height, width = frame_shape[:2]
    return OcclusionRegion(
        x_min=int(x_fraction_range[0] * width),
        x_max=int(x_fraction_range[1] * width),
        y_min=int(y_fraction_range[0] * height),
        y_max=int(y_fraction_range[1] * height),
    )


def region_around_point(
    frame_shape: tuple[int, int], center_x: float, center_y: float, radius_px: int
) -> OcclusionRegion:
    """Define una región cuadrada centrada en un keypoint específico (ej. una rodilla)."""
    height, width = frame_shape[:2]
    return OcclusionRegion(
        x_min=max(0, int(center_x - radius_px)),
        x_max=min(width, int(center_x + radius_px)),
        y_min=max(0, int(center_y - radius_px)),
        y_max=min(height, int(center_y + radius_px)),
    )


def apply_occlusion(
    frame: np.ndarray, region: OcclusionRegion, method: OcclusionMethod = OcclusionMethod.BLACK_BOX
) -> np.ndarray:
    """Devuelve una copia del frame con la región indicada ocluida."""
    occluded = frame.copy()

    if method == OcclusionMethod.BLACK_BOX:
        occluded[region.y_min:region.y_max, region.x_min:region.x_max] = 0

    elif method == OcclusionMethod.GAUSSIAN_BLUR:
        roi = occluded[region.y_min:region.y_max, region.x_min:region.x_max]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (51, 51), sigmaX=25)
            occluded[region.y_min:region.y_max, region.x_min:region.x_max] = blurred

    return occluded


def get_first_frame_shape(video_path: str) -> tuple[int, int]:
    """Devuelve (height, width) del primer frame legible del video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"No se pudo leer el video: {video_path}")
    return frame.shape[:2]


def detect_reference_joint_position(
    video_path: str, target: UnifiedKeypoint, max_probe_frames: int = 30
) -> tuple[float, float]:
    """
    Corre MediaPipe frame a frame (barato, sin guardar nada) hasta encontrar
    uno donde `target` esté visible con confianza suficiente, y devuelve su
    posición en píxeles (x, y).

    Pensado para videos donde el sujeto no se desplaza mucho por el frame
    (caminadora, ejercicios de pie/sentado en un punto fijo): UNA posición
    de referencia sirve para ocluir esa zona durante todo el video. Para
    videos con desplazamiento real (ej. caminar por un pasillo), esta
    aproximación estática no sería suficiente -- habría que ocluir por
    frame según el keypoint detectado en cada uno, algo que no se
    implementa todavía.

    Usada tanto por scripts/run_single_video.py (demo visual) como por
    scripts/evaluate_model_variants.py (--occlude-joint, métricas contra
    gold standard) para no duplicar esta lógica.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    with MediaPipePoseEstimator() as estimator:
        try:
            for _ in range(max_probe_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                for kp in estimator.predict(frame):
                    if kp.name == target and kp.visible and kp.confidence >= 0.5:
                        return kp.x, kp.y
        finally:
            cap.release()

    raise RuntimeError(
        f"No se pudo detectar {target.value} con confianza suficiente en los primeros "
        f"{max_probe_frames} frames de {video_path}. Revisa que el video muestre claramente "
        "esa articulación al inicio."
    )


def detect_reference_knee_position(video_path: str, leg: str, max_probe_frames: int = 30) -> tuple[float, float]:
    """Alias retrocompatible de detect_reference_joint_position para la rodilla."""
    target = UnifiedKeypoint.LEFT_KNEE if leg == "left" else UnifiedKeypoint.RIGHT_KNEE
    return detect_reference_joint_position(video_path, target, max_probe_frames)


def occlusion_fraction_of_frame(region: OcclusionRegion, frame_shape: tuple[int, int]) -> float:
    """
    Calcula qué porcentaje del frame ocupa la región ocluida. Útil para
    reportar niveles de oclusión de forma cuantitativa (ej. "oclusión ~15%
    del frame") en vez de solo describirla cualitativamente.
    """
    height, width = frame_shape[:2]
    total_area = height * width
    region_area = max(0, region.x_max - region.x_min) * max(0, region.y_max - region.y_min)
    return region_area / total_area if total_area > 0 else 0.0
