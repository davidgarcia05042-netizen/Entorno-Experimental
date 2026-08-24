"""
Métricas de evaluación comparativa entre modelos de pose y ground truth.

Todas las funciones reciben datos ya en el esquema unificado
(ver app/core/keypoint_schema.py), así que funcionan igual sin importar
si el origen es MediaPipe, YOLOv8, o el ground truth de laboratorio
(una vez que exista el parser de la Fase 5 que lo convierta a este mismo
formato).
"""

import math

from app.config import (
    ANGULAR_ERROR_ACCEPTABLE_DEG,
    ANGULAR_ERROR_MODERATE_DEG,
    PCK_THRESHOLD_FRACTION_OF_TORSO,
    classify_angular_error,
)
from app.core.keypoint_schema import JOINT_ANGLE_TRIPLETS, Keypoint, UnifiedKeypoint

KeypointsByName = dict[UnifiedKeypoint, Keypoint]


def _to_dict(keypoints: list[Keypoint]) -> KeypointsByName:
    return {kp.name: kp for kp in keypoints}


def euclidean_error(pred: Keypoint, gt: Keypoint) -> float:
    """Distancia euclidiana en píxeles entre un keypoint predicho y su referencia."""
    return math.hypot(pred.x - gt.x, pred.y - gt.y)


def mpjpe(pred: list[Keypoint], gt: list[Keypoint], only_visible_gt: bool = True) -> float:
    """
    Mean Per Joint Position Error: promedio del error euclidiano sobre
    todos los keypoints comparables.

    only_visible_gt: si True, ignora en el promedio los puntos donde el
    ground truth mismo no es visible/confiable (evita penalizar al modelo
    por un punto que ni el laboratorio pudo medir bien).
    """
    pred_by_name = _to_dict(pred)
    gt_by_name = _to_dict(gt)

    errors = []
    for name, gt_kp in gt_by_name.items():
        if only_visible_gt and not gt_kp.visible:
            continue
        if name not in pred_by_name:
            continue
        errors.append(euclidean_error(pred_by_name[name], gt_kp))

    if not errors:
        return float("nan")
    return sum(errors) / len(errors)


def _angle_deg(a: Keypoint, vertex: Keypoint, c: Keypoint) -> float:
    """Ángulo en grados formado en `vertex` por los segmentos vertex->a y vertex->c."""
    v1 = (a.x - vertex.x, a.y - vertex.y)
    v2 = (c.x - vertex.x, c.y - vertex.y)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return float("nan")

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles(keypoints: list[Keypoint]) -> dict[str, float]:
    """Calcula todos los ángulos articulares definidos en JOINT_ANGLE_TRIPLETS."""
    by_name = _to_dict(keypoints)
    angles = {}

    for joint_label, (p_a, p_vertex, p_c) in JOINT_ANGLE_TRIPLETS.items():
        if p_a not in by_name or p_vertex not in by_name or p_c not in by_name:
            angles[joint_label] = float("nan")
            continue
        angles[joint_label] = _angle_deg(by_name[p_a], by_name[p_vertex], by_name[p_c])

    return angles


def angular_error_report(pred: list[Keypoint], gt: list[Keypoint]) -> dict[str, dict]:
    """
    Compara los ángulos articulares entre predicción y ground truth,
    devolviendo el error absoluto en grados y su clasificación
    (aceptable / moderado / no_aceptable) por cada articulación.
    """
    pred_angles = compute_joint_angles(pred)
    gt_angles = compute_joint_angles(gt)

    report = {}
    for joint_label in JOINT_ANGLE_TRIPLETS:
        pred_angle = pred_angles[joint_label]
        gt_angle = gt_angles[joint_label]

        if math.isnan(pred_angle) or math.isnan(gt_angle):
            report[joint_label] = {"error_deg": None, "classification": "no_calculable"}
            continue

        error = abs(pred_angle - gt_angle)
        report[joint_label] = {
            "error_deg": round(error, 2),
            "classification": classify_angular_error(error),
        }

    return report


def _torso_size(keypoints: KeypointsByName) -> float | None:
    """Distancia hombro-cadera (lado derecho), usada para normalizar PCK."""
    if UnifiedKeypoint.RIGHT_SHOULDER not in keypoints or UnifiedKeypoint.RIGHT_HIP not in keypoints:
        return None
    shoulder = keypoints[UnifiedKeypoint.RIGHT_SHOULDER]
    hip = keypoints[UnifiedKeypoint.RIGHT_HIP]
    return math.hypot(shoulder.x - hip.x, shoulder.y - hip.y)


def pck(
    pred: list[Keypoint],
    gt: list[Keypoint],
    threshold_fraction: float = PCK_THRESHOLD_FRACTION_OF_TORSO,
) -> float:
    """
    Percentage of Correct Keypoints: fracción de keypoints cuyo error está
    dentro de `threshold_fraction * tamaño_del_torso`. Devuelve un valor
    entre 0.0 y 1.0.
    """
    pred_by_name = _to_dict(pred)
    gt_by_name = _to_dict(gt)

    torso = _torso_size(gt_by_name)
    if torso is None or torso == 0:
        return float("nan")

    threshold_px = threshold_fraction * torso

    correct = 0
    total = 0
    for name, gt_kp in gt_by_name.items():
        if not gt_kp.visible or name not in pred_by_name:
            continue
        total += 1
        if euclidean_error(pred_by_name[name], gt_kp) <= threshold_px:
            correct += 1

    return correct / total if total > 0 else float("nan")


def detection_rate(
    predictions: list[list[Keypoint]],
    occluded_flags: list[bool],
) -> dict[str, float]:
    """
    Tasa de detección exitosa (confidence >= umbral, ver config.py) separada
    entre frames marcados como ocluidos y no ocluidos. Esta separación es
    el corazón de la pregunta de investigación: no basta con una tasa
    global, hay que ver la degradación específica bajo oclusión.

    `occluded_flags` debe tener la misma longitud que `predictions`, marcando
    con True los frames donde hubo oclusión (natural o programada).
    """
    from app.config import DETECTION_CONFIDENCE_THRESHOLD

    def _mean_detection(frames: list[list[Keypoint]]) -> float:
        if not frames:
            return float("nan")
        rates = []
        for kpts in frames:
            detected = sum(1 for kp in kpts if kp.confidence >= DETECTION_CONFIDENCE_THRESHOLD)
            rates.append(detected / len(kpts) if kpts else 0.0)
        return sum(rates) / len(rates)

    occluded_frames = [p for p, flag in zip(predictions, occluded_flags) if flag]
    visible_frames = [p for p, flag in zip(predictions, occluded_flags) if not flag]

    return {
        "detection_rate_occluded": _mean_detection(occluded_frames),
        "detection_rate_visible": _mean_detection(visible_frames),
        "n_occluded_frames": len(occluded_frames),
        "n_visible_frames": len(visible_frames),
    }
