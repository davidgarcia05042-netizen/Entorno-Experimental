"""
Esquema unificado de keypoints.

Problema que resuelve este módulo
----------------------------------
MediaPipe Pose y YOLOv8-Pose NO usan el mismo esquema de puntos anatómicos:

- MediaPipe Pose  -> 33 landmarks (incluye rostro detallado, manos, talones, pies)
- YOLOv8-Pose      -> 17 keypoints, formato COCO (el estándar de detección de personas)

Para poder comparar ambos modelos entre sí, y contra el ground truth de
laboratorio (marcadores físicos en articulaciones clave), necesitamos un
"idioma común". Este módulo define ese vocabulario común usando los nombres
COCO (17 puntos) como base, porque:

  1. Es el subconjunto que ambos modelos pueden producir.
  2. Cubre las articulaciones relevantes para fisioterapia (hombros, codos,
     muñecas, caderas, rodillas, tobillos).
  3. Es el estándar más usado en la literatura de pose estimation, lo que
     facilita comparar tus resultados con otros trabajos citados en el
     estado del arte.

IMPORTANTE: Los índices de MediaPipe listados abajo corresponden a la
documentación pública de `mediapipe.solutions.pose.PoseLandmark`. Si migran
a la nueva Tasks API de MediaPipe (`PoseLandmarker`), confirmen que el orden
de los landmarks no cambió antes de usar este mapeo en producción.
"""

from dataclasses import dataclass
from enum import Enum


class UnifiedKeypoint(str, Enum):
    """Vocabulario común de 17 puntos (formato COCO) usado en todo el servicio."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


# Orden oficial COCO (el que produce YOLOv8-Pose de Ultralytics de forma nativa)
COCO_ORDER = [
    UnifiedKeypoint.NOSE,
    UnifiedKeypoint.LEFT_EYE,
    UnifiedKeypoint.RIGHT_EYE,
    UnifiedKeypoint.LEFT_EAR,
    UnifiedKeypoint.RIGHT_EAR,
    UnifiedKeypoint.LEFT_SHOULDER,
    UnifiedKeypoint.RIGHT_SHOULDER,
    UnifiedKeypoint.LEFT_ELBOW,
    UnifiedKeypoint.RIGHT_ELBOW,
    UnifiedKeypoint.LEFT_WRIST,
    UnifiedKeypoint.RIGHT_WRIST,
    UnifiedKeypoint.LEFT_HIP,
    UnifiedKeypoint.RIGHT_HIP,
    UnifiedKeypoint.LEFT_KNEE,
    UnifiedKeypoint.RIGHT_KNEE,
    UnifiedKeypoint.LEFT_ANKLE,
    UnifiedKeypoint.RIGHT_ANKLE,
]

# Mapeo: nombre unificado -> índice del landmark en MediaPipe Pose (0-32)
MEDIAPIPE_INDEX = {
    UnifiedKeypoint.NOSE: 0,
    UnifiedKeypoint.LEFT_EYE: 2,
    UnifiedKeypoint.RIGHT_EYE: 5,
    UnifiedKeypoint.LEFT_EAR: 7,
    UnifiedKeypoint.RIGHT_EAR: 8,
    UnifiedKeypoint.LEFT_SHOULDER: 11,
    UnifiedKeypoint.RIGHT_SHOULDER: 12,
    UnifiedKeypoint.LEFT_ELBOW: 13,
    UnifiedKeypoint.RIGHT_ELBOW: 14,
    UnifiedKeypoint.LEFT_WRIST: 15,
    UnifiedKeypoint.RIGHT_WRIST: 16,
    UnifiedKeypoint.LEFT_HIP: 23,
    UnifiedKeypoint.RIGHT_HIP: 24,
    UnifiedKeypoint.LEFT_KNEE: 25,
    UnifiedKeypoint.RIGHT_KNEE: 26,
    UnifiedKeypoint.LEFT_ANKLE: 27,
    UnifiedKeypoint.RIGHT_ANKLE: 28,
}

# Mapeo: nombre unificado -> índice en la salida COCO-17 de YOLOv8-Pose
# (Es simplemente la posición en COCO_ORDER, pero se deja explícito para
#  que quede documentado y no se rompa si cambian el orden en el futuro)
YOLO_COCO_INDEX = {kp: i for i, kp in enumerate(COCO_ORDER)}

# Tripletas (proximal, vértice, distal) para calcular ángulos articulares.
# El ángulo se calcula EN el punto del medio (vértice).
JOINT_ANGLE_TRIPLETS = {
    "left_elbow": (UnifiedKeypoint.LEFT_SHOULDER, UnifiedKeypoint.LEFT_ELBOW, UnifiedKeypoint.LEFT_WRIST),
    "right_elbow": (UnifiedKeypoint.RIGHT_SHOULDER, UnifiedKeypoint.RIGHT_ELBOW, UnifiedKeypoint.RIGHT_WRIST),
    "left_knee": (UnifiedKeypoint.LEFT_HIP, UnifiedKeypoint.LEFT_KNEE, UnifiedKeypoint.LEFT_ANKLE),
    "right_knee": (UnifiedKeypoint.RIGHT_HIP, UnifiedKeypoint.RIGHT_KNEE, UnifiedKeypoint.RIGHT_ANKLE),
    "left_shoulder": (UnifiedKeypoint.LEFT_HIP, UnifiedKeypoint.LEFT_SHOULDER, UnifiedKeypoint.LEFT_ELBOW),
    "right_shoulder": (UnifiedKeypoint.RIGHT_HIP, UnifiedKeypoint.RIGHT_SHOULDER, UnifiedKeypoint.RIGHT_ELBOW),
    "left_hip": (UnifiedKeypoint.LEFT_SHOULDER, UnifiedKeypoint.LEFT_HIP, UnifiedKeypoint.LEFT_KNEE),
    "right_hip": (UnifiedKeypoint.RIGHT_SHOULDER, UnifiedKeypoint.RIGHT_HIP, UnifiedKeypoint.RIGHT_KNEE),
}


# Pares de puntos que se conectan al dibujar el esqueleto sobre el video.
# Es el set de conexiones estándar del formato COCO-17.
SKELETON_EDGES = [
    (UnifiedKeypoint.NOSE, UnifiedKeypoint.LEFT_EYE),
    (UnifiedKeypoint.NOSE, UnifiedKeypoint.RIGHT_EYE),
    (UnifiedKeypoint.LEFT_EYE, UnifiedKeypoint.LEFT_EAR),
    (UnifiedKeypoint.RIGHT_EYE, UnifiedKeypoint.RIGHT_EAR),
    (UnifiedKeypoint.LEFT_SHOULDER, UnifiedKeypoint.RIGHT_SHOULDER),
    (UnifiedKeypoint.LEFT_SHOULDER, UnifiedKeypoint.LEFT_ELBOW),
    (UnifiedKeypoint.LEFT_ELBOW, UnifiedKeypoint.LEFT_WRIST),
    (UnifiedKeypoint.RIGHT_SHOULDER, UnifiedKeypoint.RIGHT_ELBOW),
    (UnifiedKeypoint.RIGHT_ELBOW, UnifiedKeypoint.RIGHT_WRIST),
    (UnifiedKeypoint.LEFT_SHOULDER, UnifiedKeypoint.LEFT_HIP),
    (UnifiedKeypoint.RIGHT_SHOULDER, UnifiedKeypoint.RIGHT_HIP),
    (UnifiedKeypoint.LEFT_HIP, UnifiedKeypoint.RIGHT_HIP),
    (UnifiedKeypoint.LEFT_HIP, UnifiedKeypoint.LEFT_KNEE),
    (UnifiedKeypoint.LEFT_KNEE, UnifiedKeypoint.LEFT_ANKLE),
    (UnifiedKeypoint.RIGHT_HIP, UnifiedKeypoint.RIGHT_KNEE),
    (UnifiedKeypoint.RIGHT_KNEE, UnifiedKeypoint.RIGHT_ANKLE),
]


@dataclass
class Keypoint:
    """Un punto anatómico ya normalizado al esquema unificado."""

    name: UnifiedKeypoint
    x: float  # coordenada horizontal en píxeles (o normalizada 0-1, ver PoseFrameResult)
    y: float  # coordenada vertical en píxeles
    confidence: float  # score de confianza del modelo, 0.0 a 1.0
    visible: bool = True  # False si el modelo marcó el punto como no visible/inferido


def todo_check_mediapipe_task_api_indices() -> None:
    """
    Recordatorio activo: si el equipo usa la nueva Tasks API de MediaPipe
    (mediapipe.tasks.python.vision.PoseLandmarker) en vez de
    mediapipe.solutions.pose, hay que confirmar que PoseLandmarker sigue
    devolviendo los 33 puntos en el mismo orden documentado aquí.
    """
    pass
