"""
Tests que NO requieren mediapipe ni ultralytics instalados (solo prueban
la lógica pura: esquema de keypoints, métricas, oclusión, iluminación).
Los wrappers de modelos (mediapipe_pose.py, yolov8_pose.py) se prueban
aparte, una vez esas librerías estén instaladas en el entorno de desarrollo.

Ejecutar con: pytest tests/test_core.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from app.config import classify_angular_error
from app.core.ground_truth import (
    MaxtraqRecording,
    compare_angle_series,
    find_best_time_offset,
    knee_angle_series_deg,
    parse_maxtraq_txt,
)
from app.core.keypoint_schema import COCO_ORDER, MEDIAPIPE_INDEX, YOLO_COCO_INDEX, Keypoint
from app.core.metrics import angular_error_report, mpjpe, pck
from app.core.occlusion import apply_occlusion, occlusion_fraction_of_frame, region_from_fixed_fraction
from app.core.platform_export import parse_platform_csv_angle_series, parse_platform_json_angle_series
from app.core.results_store import create_run_dir, save_frame_metrics_csv, save_keypoints_json, save_summary_csv
from app.schemas.pose_result import ModelName, VideoAnalysisResult


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


_SAMPLE_MAXTRAQ_TXT = """Frame number,3
First frame,1
Point frequency,60
Analog frequency,60

Time,Point #1,,,Point #2,,,Point #3,,,Point #4,,,Point #5,,,Point #6,,
s,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm
,X,Y,Z,X,Y,Z,X,Y,Z,X,Y,Z,X,Y,Z,X,Y,Z
0,0,0,0,0,0,0,0,10,0,0,0,0,0,20,0,0,0,0
0.0166667,0,0,0,0,0,0,0,10,0,0,0,0,0,20,0,0,0,0
0.0333333,,,,0,0,0,0,10,0,0,0,0,0,20,0,0,0,0
"""


def test_parse_maxtraq_txt_reads_header_and_points(tmp_path: Path):
    path = tmp_path / "sample_maxtraq.TXT"
    path.write_text(_SAMPLE_MAXTRAQ_TXT, encoding="utf-8")

    recording = parse_maxtraq_txt(path)

    assert recording.point_frequency_hz == 60.0
    assert recording.total_frames == 3
    assert len(recording.times_s) == 3
    assert set(recording.points.keys()) == {1, 2, 3, 4, 5, 6}
    assert recording.points[3][0] == (0.0, 10.0, 0.0)


def test_parse_maxtraq_txt_marks_missing_marker_as_none(tmp_path: Path):
    path = tmp_path / "sample_maxtraq.TXT"
    path.write_text(_SAMPLE_MAXTRAQ_TXT, encoding="utf-8")

    recording = parse_maxtraq_txt(path)

    assert recording.points[1][2] is None  # tercer frame, marcador #1 vacío


def test_knee_angle_series_deg_computes_straight_leg_as_180():
    # Marcadores 1 (cadera), 3 (rodilla), 5 (tobillo) alineados -> pierna
    # recta -> 180°.
    recording = MaxtraqRecording(
        point_frequency_hz=60.0,
        total_frames=1,
        times_s=[0.0],
        points={1: [(0.0, 0.0, 0.0)], 3: [(0.0, 10.0, 0.0)], 5: [(0.0, 20.0, 0.0)]},
    )

    angles = knee_angle_series_deg(recording)
    assert angles[0] == pytest.approx(180.0)


def test_compare_angle_series_zero_error_for_identical_series():
    result = compare_angle_series([170.0, 165.0, 180.0], [170.0, 165.0, 180.0])
    assert result["mean_error_deg"] == 0.0
    assert result["classification"] == "aceptable"


def test_find_best_time_offset_recovers_known_shift():
    import math

    # Señal periódica sintética (simula un ciclo de marcha) en el "reloj" del gt.
    gt_times = [i * 0.1 for i in range(200)]  # 0 a 19.9s
    gt_angles = [150 + 30 * math.sin(t) for t in gt_times]

    # La misma señal, pero el reloj del "query" arrancó 2.5s despues del gt.
    known_offset = 2.5
    query_times = [t + known_offset for t in gt_times]
    query_angles = gt_angles

    result = find_best_time_offset(gt_times, gt_angles, query_times, query_angles, search_range_s=(-5.0, 5.0))

    assert result["offset_s"] == pytest.approx(known_offset, abs=0.05)
    assert result["correlation"] > 0.99


_SAMPLE_PLATFORM_CSV = (
    "Tiempo (s),Cadera Izquierda X,Cadera Izquierda Y,Cadera Izquierda Ángulo (°),"
    "Tobillo Izquierdo X,Tobillo Izquierdo Y,Tobillo Izquierdo Ángulo (°)\n"
    "0.000,0.51,0.42,98.5,0.50,0.68,\n"
    "0.033,0.51,0.43,93.1,0.50,0.68,\n"
)


def test_parse_platform_csv_angle_series_skips_empty_column(tmp_path: Path):
    path = tmp_path / "platform.csv"
    path.write_text(_SAMPLE_PLATFORM_CSV, encoding="utf-8")

    times, angles = parse_platform_csv_angle_series(path, column_name="Cadera Izquierda Ángulo (°)")
    assert times == [0.0, 0.033]
    assert angles == [98.5, 93.1]

    # La columna del tobillo está vacía en ambas filas -> no debe devolver nada.
    empty_times, empty_angles = parse_platform_csv_angle_series(path, column_name="Tobillo Izquierdo Ángulo (°)")
    assert empty_times == []
    assert empty_angles == []


_SAMPLE_PLATFORM_JSON = """{
  "exercise_id": 1,
  "exercise_name": "Test",
  "duration": 0.033,
  "frames": [
    {"t": 0, "points": [], "angles": {"LEFT_HIP": 98.5, "LEFT_KNEE": 157.6}},
    {"t": 0.033, "points": [], "angles": {"LEFT_HIP": 93.1}}
  ]
}"""


def test_parse_platform_json_angle_series_skips_frames_missing_codename(tmp_path: Path):
    path = tmp_path / "platform.json"
    path.write_text(_SAMPLE_PLATFORM_JSON, encoding="utf-8")

    times, angles = parse_platform_json_angle_series(path, codename="LEFT_KNEE")
    assert times == [0.0]
    assert angles == [157.6]


def test_create_run_dir_makes_a_timestamped_folder_under_runs(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)

    assert run_dir.is_dir()
    assert run_dir.parent.name == "runs"
    assert run_dir.parent.parent == tmp_path


def test_save_keypoints_json_writes_readable_pose_result(tmp_path: Path):
    result = VideoAnalysisResult(
        video_id="test_video",
        model_name=ModelName.YOLOV8,
        fps=30.0,
        total_frames=0,
        frames=[],
        mean_latency_ms=0.0,
    )
    run_dir = create_run_dir(tmp_path)

    out_path = save_keypoints_json(run_dir, "yolo26n", result)

    assert out_path.exists()
    assert '"video_id":"test_video"' in out_path.read_text(encoding="utf-8").replace(" ", "").replace("\n", "")


def test_save_frame_metrics_csv_includes_error_per_model(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)
    gt_times = [0.0, 1.0]
    gt_angles = [180.0, 170.0]
    model_curves = {"yolo26n": [175.0, 168.0], "mediapipe_heavy": [178.0, 172.0]}

    out_path = save_frame_metrics_csv(run_dir, gt_times, gt_angles, model_curves, leg="left_knee")

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "frame_idx,time_s,gt_left_knee_angle_deg,yolo26n_angle_deg,yolo26n_error_deg,mediapipe_heavy_angle_deg,mediapipe_heavy_error_deg"
    assert lines[1] == "0,0.0,180.0,175.0,5.0,178.0,2.0"


def test_save_summary_csv_writes_one_row_per_variant(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)
    reports = {
        "yolo26n": {"mean_error_deg": 8.45, "max_error_deg": 20.1, "n_frames": 333, "classification": "moderado", "elapsed_s": 19.5},
    }

    out_path = save_summary_csv(run_dir, reports)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "variante,error_medio_deg,error_max_deg,n_frames,clasificacion,tiempo_s"
    assert lines[1] == "yolo26n,8.45,20.1,333,moderado,19.5"
