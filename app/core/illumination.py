"""
Simulación de variación de iluminación (plan de respaldo).

IMPORTANTE - limitación metodológica a documentar en el trabajo de grado:
Este módulo simula variación de LUMINOSIDAD en post-procesamiento (ajuste
de brillo/gamma sobre los píxeles ya capturados). Esto NO es equivalente a
variar la iluminación física real de la escena: no genera sombras nuevas,
no replica el comportamiento del sensor de la cámara ante distinta luz
(exposición automática, ruido en baja luz), y no corresponde a un valor de
lux medido con instrumento. Es una aproximación común en estudios de
robustez de visión por computador, pero debe describirse explícitamente
como "variación simulada de brillo", no como "variación de iluminación
real medida en lux", si finalmente no se logra la grabación real en
laboratorio.
"""

from enum import Enum

import cv2
import numpy as np


class IlluminationLevel(str, Enum):
    BRIGHT = "bright"       # simula sobreexposición / luz muy fuerte
    NORMAL = "normal"       # sin modificación, línea base
    DIM = "dim"              # simula iluminación doméstica tenue (~100-150 lux)
    DARK = "dark"            # simula baja luz, cerca del umbral crítico reportado
                              # en literatura para pose estimation (~15-20 lux)


# Factores de ganancia gamma por nivel. gamma < 1 aclara, gamma > 1 oscurece.
# Calibrar estos valores con pruebas visuales antes de usarlos en el
# experimento formal: el objetivo es que "DARK" luzca perceptualmente
# comparable a una habitación con luz ambiente muy baja, no simplemente
# "una imagen oscura sin sentido visual".
_GAMMA_BY_LEVEL = {
    IlluminationLevel.BRIGHT: 0.6,
    IlluminationLevel.NORMAL: 1.0,
    IlluminationLevel.DIM: 1.8,
    IlluminationLevel.DARK: 3.0,
}


def apply_gamma_correction(frame: np.ndarray, gamma: float) -> np.ndarray:
    """
    Aplica corrección gamma: salida = 255 * (entrada/255) ** gamma.
    gamma > 1 oscurece (curva cóncava, empuja los valores hacia 0);
    gamma < 1 aclara (curva convexa, empuja los valores hacia 255).
    """
    if gamma == 1.0:
        return frame.copy()

    lookup_table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(frame, lookup_table)


def simulate_illumination(frame: np.ndarray, level: IlluminationLevel) -> np.ndarray:
    """Devuelve una copia del frame con el nivel de iluminación simulado."""
    gamma = _GAMMA_BY_LEVEL[level]
    return apply_gamma_correction(frame, gamma)


def generate_illumination_variants(frame: np.ndarray) -> dict[IlluminationLevel, np.ndarray]:
    """Genera las 4 variantes de un mismo frame, una por cada nivel definido."""
    return {level: simulate_illumination(frame, level) for level in IlluminationLevel}


def estimate_mean_brightness(frame: np.ndarray) -> float:
    """
    Brillo promedio del frame en escala de grises (0-255), útil como proxy
    RELATIVO para reportar cuánto cambió una variante respecto a otra.
    No es una medición de lux: es una lectura de valores de píxel, no de
    luz incidente real captada con luxómetro.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
