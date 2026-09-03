"""
Diagnóstico puntual: identificar qué articulación representan los
marcadores #1/#2/#3 de 2EJ1.TXT ("Levantamiento de BRAZOS"), probando
las 3 asignaciones posibles de vértice contra el ángulo de hombro/codo
(izq. y der.) que predice MediaPipe sobre el mismo video, vía correlación
cruzada -- mismo método usado para confirmar la rodilla de Marcha
Katherine (ver app/core/ground_truth.py::find_best_time_offset).

No es parte del pipeline principal, solo para decidir la asignación de
marcadores antes de generalizar knee_angle_series_deg.
"""

from app.core.ground_truth import _angle_3d_deg, find_best_time_offset, parse_maxtraq_txt
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.schemas.pose_result import ModelName

VIDEO = "data/gold_standard/TOMA FRONTAL/2EJ1/2EJ1.mp4"
TXT = "data/gold_standard/TOMA FRONTAL/2EJ1/2EJ1.TXT"

VERTEX_ORDER = {1: (2, 1, 3), 2: (1, 2, 3), 3: (1, 3, 2)}


def _marker_angle_series(points_by_id: dict[int, list], vertex_id: int) -> list[float]:
    a_id, v_id, c_id = VERTEX_ORDER[vertex_id]
    a_pts, v_pts, c_pts = points_by_id[a_id], points_by_id[v_id], points_by_id[c_id]
    out = []
    for a, v, c in zip(a_pts, v_pts, c_pts):
        out.append(float("nan") if None in (a, v, c) else _angle_3d_deg(a, v, c))
    return out


def main() -> None:
    recording = parse_maxtraq_txt(TXT)
    gt_times = recording.times_s
    marker_candidates = {f"vertice_{i}": _marker_angle_series(recording.points, i) for i in (1, 2, 3)}

    print(f"Corriendo MediaPipe sobre {VIDEO}...")
    with MediaPipePoseEstimator() as estimator:
        result = process_video(VIDEO, "diag_2ej1", estimator, ModelName.MEDIAPIPE)

    times_s = [f.timestamp_ms / 1000 for f in result.frames]
    joint_candidates: dict[str, list[float]] = {
        joint: [] for joint in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
    }
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        angles = compute_joint_angles(kpts)
        for joint in joint_candidates:
            joint_candidates[joint].append(angles[joint])

    print()
    print(f"{'marcador':10s} {'articulacion':15s} {'offset_s':>9s} {'corr':>7s}")
    print("-" * 45)
    best = None
    for m_label, m_series in marker_candidates.items():
        for j_label, j_series in joint_candidates.items():
            res = find_best_time_offset(
                gt_times, m_series, times_s, j_series, search_range_s=(-5.0, 15.0), min_overlap_fraction=0.5
            )
            print(f"{m_label:10s} {j_label:15s} {res['offset_s']:9.2f} {res['correlation']:7.3f}")
            if best is None or res["correlation"] > best[2]:
                best = (m_label, j_label, res["correlation"], res["offset_s"])

    print()
    print(f"Mejor combinación: marcador vértice={best[0]}, articulación MediaPipe={best[1]}, "
          f"corr={best[2]:.3f}, offset={best[3]:.2f}s")


if __name__ == "__main__":
    main()
