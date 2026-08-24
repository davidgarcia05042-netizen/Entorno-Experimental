"""
Tests que NO requieren mediapipe ni ultralytics instalados (solo prueban
la lógica pura: esquema de keypoints, métricas, oclusión, iluminación).
Los wrappers de modelos (mediapipe_pose.py, yolov8_pose.py) se prueban
aparte, una vez esas librerías estén instaladas en el entorno de desarrollo.

Ejecutar con: pytest tests/test_core.py -v
"""

import numpy as np

from app.config import classify_angular_error
from app.core.keypoint_schema import COCO_ORDER, MEDIAPIPE_INDEX, YOLO_COCO_INDEX, Keypoint
from app.core.metrics import angular_error_report, mpjpe, pck
from app.core.occlusion import apply_occlusion, occlusion_fraction_of_frame, region_from_fixed_fraction


def test_keypoint_schema_has_all_17_coco_points():
    assert len(COCO_ORDER) == 17
    assert len(MEDIAPIPE_INDEX) == 17
    assert len(YOLO_COCO_INDEX) == 17
    assert set(MEDIAPIPE_INDEX.keys()) == set(COCO_ORDER)


def test_mediapipe_indices_are_within_valid_range():
    # MediaPipe Pose tiene 33 landmarks válidos: índices 0 a 32
    for name, idx in MEDIAPIPE_INDEX.items():
        assert 0 <= idx <= 32, f"{name} tiene índice inválido: {idx}"


def _make_keypoints(offset: float = 0.0) -> list[Keypoint]:
    """Genera un set de keypoints sintético para pruebas, en línea recta."""
    return [
        Keypoint(name=name, x=10.0 * i + offset, y=10.0 * i + offset, confidence=0.9, visible=True)
        for i, name in enumerate(COCO_ORDER)
    ]


def test_mpjpe_is_zero_for_identical_keypoints():
    kpts = _make_keypoints()
    assert mpjpe(kpts, kpts) == 0.0


def test_mpjpe_increases_with_offset():
    gt = _make_keypoints(offset=0.0)
    pred = _make_keypoints(offset=5.0)
    error = mpjpe(pred, gt)
    assert error > 0.0


def test_angular_error_classification_thresholds():
    assert classify_angular_error(3.0) == "aceptable"
    assert classify_angular_error(5.0) == "aceptable"
    assert classify_angular_error(7.5) == "moderado"
    assert classify_angular_error(15.0) == "no_aceptable"


def test_angular_error_report_returns_all_joints():
    kpts = _make_keypoints()
    report = angular_error_report(kpts, kpts)
    assert "left_knee" in report
    assert "right_elbow" in report
    # Mismos puntos -> mismo ángulo -> error 0 -> aceptable
    for joint_report in report.values():
        if joint_report["error_deg"] is not None:
            assert joint_report["classification"] == "aceptable"


def test_pck_perfect_match_is_one():
    kpts = _make_keypoints()
    score = pck(kpts, kpts)
    assert score == 1.0


def test_occlusion_region_fraction_is_reasonable():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    region = region_from_fixed_fraction(frame.shape, x_fraction_range=(0.0, 1.0), y_fraction_range=(0.66, 1.0))
    fraction = occlusion_fraction_of_frame(region, frame.shape)
    assert 0.3 < fraction < 0.4  # ~ un tercio del frame


def test_apply_occlusion_blackens_region():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)  # frame blanco
    region = region_from_fixed_fraction(frame.shape, (0.0, 0.5), (0.0, 0.5))
    occluded = apply_occlusion(frame, region)
    assert occluded[10, 10].sum() == 0  # dentro de la región: negro
    assert occluded[90, 90].sum() == 255 * 3  # fuera de la región: intacto
