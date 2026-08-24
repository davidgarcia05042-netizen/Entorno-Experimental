"""
Visualización: dibuja los keypoints y el esqueleto detectados sobre un frame.

El color de cada punto refleja la confianza del modelo, no solo su posición:
  - Verde: confianza >= umbral (detección confiable)
  - Rojo:  confianza < umbral (detección dudosa o inferida bajo oclusión)

Esto es deliberado: para su pregunta de investigación (comportamiento bajo
oclusión), ver EN el video dónde el modelo empieza a "adivinar" es más
revelador que solo mirar el número de confianza en el JSON.
"""

import cv2
import numpy as np

from app.core.keypoint_schema import SKELETON_EDGES, Keypoint

# Colores en formato BGR (el que usa OpenCV, no RGB)
COLOR_HIGH_CONFIDENCE = (0, 200, 0)     # verde
COLOR_LOW_CONFIDENCE = (0, 0, 220)      # rojo
COLOR_SKELETON_LINE = (0, 165, 255)     # naranja


def draw_pose_on_frame(
    frame: np.ndarray,
    keypoints: list[Keypoint],
    confidence_threshold: float = 0.5,
    point_radius: int = 5,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Devuelve una COPIA del frame con el esqueleto dibujado encima.
    No modifica el frame original (importante: process_video sigue
    necesitando el frame limpio para otros usos si se reprocesa).
    """
    annotated = frame.copy()
    by_name = {kp.name: kp for kp in keypoints}

    # 1. Líneas del esqueleto primero (para que los puntos queden encima)
    for point_a, point_b in SKELETON_EDGES:
        kp_a = by_name.get(point_a)
        kp_b = by_name.get(point_b)
        if kp_a is None or kp_b is None:
            continue
        if not kp_a.visible or not kp_b.visible:
            continue
        cv2.line(
            annotated,
            (int(kp_a.x), int(kp_a.y)),
            (int(kp_b.x), int(kp_b.y)),
            COLOR_SKELETON_LINE,
            line_thickness,
        )

    # 2. Puntos encima de las líneas, coloreados por confianza
    for kp in keypoints:
        if not kp.visible:
            continue
        color = COLOR_HIGH_CONFIDENCE if kp.confidence >= confidence_threshold else COLOR_LOW_CONFIDENCE
        cv2.circle(annotated, (int(kp.x), int(kp.y)), point_radius, color, thickness=-1)

    return annotated


def draw_frame_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Agrega una etiqueta de texto simple en la esquina superior izquierda (ej. nombre del modelo)."""
    annotated = frame.copy()
    cv2.putText(
        annotated,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
