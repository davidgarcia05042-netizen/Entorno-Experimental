"""
Configuración y constantes del servicio.

Los umbrales de error angular están fundamentados en:
- Gajdosik & Bohannon (1988), estándar clásico de goniometría clínica:
  <=5 deg aceptable, 5-10 deg moderado, >10 deg pobre.
- Estudios de validación de mocap markerless vs. marker-based que reportan
  MAE < 5 deg como rango aceptable para análisis biomecánico clínico.

Cítenlos con la referencia bibliográfica completa en el marco teórico,
no solo en el código.
"""

# --- Umbrales de error angular (grados) ---
ANGULAR_ERROR_ACCEPTABLE_DEG = 5.0
ANGULAR_ERROR_MODERATE_DEG = 10.0
# > ANGULAR_ERROR_MODERATE_DEG se considera "no aceptable"

# --- Umbrales de iluminación simulada (ver app/core/illumination.py) ---
# Referencia aproximada de lux que estos niveles buscan aproximar
# (ver conversación de diseño metodológico, no son valores medidos):
ILLUMINATION_LUX_REFERENCE = {
    "bright": 500,
    "normal": 300,
    "dim": 120,
    "dark": 20,
}

# --- PCK (Percentage of Correct Keypoints) ---
# Umbral de distancia normalizado por el tamaño del torso del sujeto
# (distancia hombro-cadera), estándar común en la literatura de pose
# estimation para que el umbral no dependa de la resolución del video.
PCK_THRESHOLD_FRACTION_OF_TORSO = 0.2

# --- Confianza mínima para considerar un keypoint como "detectado" ---
DETECTION_CONFIDENCE_THRESHOLD = 0.5


def classify_angular_error(error_deg: float) -> str:
    """Clasifica un error angular según los umbrales definidos arriba."""
    if error_deg <= ANGULAR_ERROR_ACCEPTABLE_DEG:
        return "aceptable"
    if error_deg <= ANGULAR_ERROR_MODERATE_DEG:
        return "moderado"
    return "no_aceptable"
