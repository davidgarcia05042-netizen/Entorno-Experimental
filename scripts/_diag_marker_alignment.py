"""
Diagnóstico puntual (generalización de _diag_2ej1_marker_alignment.py):
identificar qué articulación representan los marcadores #1/#2/#3 en cada
categoría de ejercicio de "TOMA FRONTAL", probando las 3 asignaciones
posibles de vértice contra 10 candidatos de articulación calculados
DIRECTAMENTE de los 33 landmarks crudos de MediaPipe (no del esquema
COCO-17 unificado, que no incluye tobillo/pie -- necesario para la
categoría "pie derecho").

No es parte del pipeline principal, solo para decidir la asignación de
marcadores antes de generalizar knee_angle_series_deg a otras
articulaciones.
"""

from app.core.ground_truth import _angle_3d_deg, find_best_time_offset, parse_maxtraq_txt

VIDEOS = [
    ("2EJ3 - antebrazo derecho", "data/gold_standard/TOMA FRONTAL/2EJ3/2EJ3.mp4", "data/gold_standard/TOMA FRONTAL/2EJ3/2EJ3.TXT"),
    ("2EJ5 - pierna derecha", "data/gold_standard/TOMA FRONTAL/2EJ5/2EJ5.mp4", "data/gold_standard/TOMA FRONTAL/2EJ5/2EJ5.TXT"),
    ("3EJ7 - pie derecho", "data/gold_standard/TOMA FRONTAL/3EJ7/3EJ7.mp4", "data/gold_standard/TOMA FRONTAL/3EJ7/3EJ7.TXT"),
    ("3EJ10 - sentadilla", "data/gold_standard/TOMA FRONTAL/3EJ10/3EJ10.mp4", "data/gold_standard/TOMA FRONTAL/3EJ10/3EJ10.TXT"),
]

VERTEX_ORDER = {1: (2, 1, 3), 2: (1, 2, 3), 3: (1, 3, 2)}

# índice de landmark crudo de MediaPipe (mp.solutions.pose.PoseLandmark)
LM = {
    "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28), "foot_index": (31, 32),
}
# (proximal, vertice, distal) por articulación, izq=0/der=1 vía LM
JOINT_TRIPLETS = {
    "shoulder": ("hip", "shoulder", "elbow"),
    "elbow": ("shoulder", "elbow", "wrist"),
    "hip": ("shoulder", "hip", "knee"),
    "knee": ("hip", "knee", "ankle"),
    "ankle": ("knee", "ankle", "foot_index"),
}


def _marker_angle_series(points_by_id: dict[int, list], vertex_id: int) -> list[float]:
    a_id, v_id, c_id = VERTEX_ORDER[vertex_id]
    a_pts, v_pts, c_pts = points_by_id[a_id], points_by_id[v_id], points_by_id[c_id]
    out = []
    for a, v, c in zip(a_pts, v_pts, c_pts):
        out.append(float("nan") if None in (a, v, c) else _angle_3d_deg(a, v, c))
    return out


def _run_mediapipe_raw_landmarks(video_path: str):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    times_s, frames_landmarks = [], []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(frame[:, :, ::-1])
            times_s.append(frame_idx / fps)
            frames_landmarks.append(results.pose_landmarks.landmark if results.pose_landmarks else None)
            frame_idx += 1
    cap.release()
    return times_s, frames_landmarks


def _joint_angle_series(frames_landmarks, joint: str, side: int) -> list[float]:
    prox_name, vertex_name, dist_name = JOINT_TRIPLETS[joint]
    prox_idx, vertex_idx, dist_idx = LM[prox_name][side], LM[vertex_name][side], LM[dist_name][side]
    out = []
    for landmarks in frames_landmarks:
        if landmarks is None:
            out.append(float("nan"))
            continue
        a = landmarks[prox_idx]
        v = landmarks[vertex_idx]
        c = landmarks[dist_idx]
        out.append(_angle_3d_deg((a.x, a.y, a.z), (v.x, v.y, v.z), (c.x, c.y, c.z)))
    return out


def main() -> None:
    for label, video_path, txt_path in VIDEOS:
        print("=" * 70)
        print(label)
        print("=" * 70)

        recording = parse_maxtraq_txt(txt_path)
        gt_times = recording.times_s
        marker_candidates = {f"vertice_{i}": _marker_angle_series(recording.points, i) for i in (1, 2, 3)}

        print(f"Corriendo MediaPipe (landmarks crudos) sobre {video_path}...")
        times_s, frames_landmarks = _run_mediapipe_raw_landmarks(video_path)

        joint_candidates: dict[str, list[float]] = {}
        for joint in JOINT_TRIPLETS:
            for side, side_label in ((0, "left"), (1, "right")):
                joint_candidates[f"{side_label}_{joint}"] = _joint_angle_series(frames_landmarks, joint, side)

        results = []
        for m_label, m_series in marker_candidates.items():
            for j_label, j_series in joint_candidates.items():
                res = find_best_time_offset(
                    gt_times, m_series, times_s, j_series, search_range_s=(-5.0, 15.0), min_overlap_fraction=0.5
                )
                results.append((m_label, j_label, res["correlation"], res["offset_s"]))

        results.sort(key=lambda r: -r[2])
        print(f"\n{'marcador':10s} {'articulacion':15s} {'offset_s':>9s} {'corr':>7s}")
        print("-" * 45)
        for m_label, j_label, corr, offset in results[:8]:
            print(f"{m_label:10s} {j_label:15s} {offset:9.2f} {corr:7.3f}")

        best = results[0]
        print(f"\nMejor combinación: marcador vértice={best[0]}, articulación MediaPipe={best[1]}, "
              f"corr={best[2]:.3f}, offset={best[3]:.2f}s\n")


if __name__ == "__main__":
    main()
