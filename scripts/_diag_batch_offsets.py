"""
Diagnóstico puntual: determina el offset de sincronización de los videos
de TOMA FRONTAL restantes (los que aún no se probaron individualmente),
usando la articulación ya asignada por categoría de ejercicio y el
marcador vértice=#2 ya confirmado como convención general.

No es parte del pipeline principal.
"""

from app.core.ground_truth import find_best_time_offset, marker_angle_series_deg, parse_maxtraq_txt
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.schemas.pose_result import ModelName

BASE = "data/gold_standard/TOMA FRONTAL"

# (video, joint) -- joint asignado por categoría (rodilla=pierna/sentadilla, codo=antebrazo)
VIDEOS = [
    ("3EJ6", "right_knee"),
    ("3EJ9", "right_knee"),
    ("4EJ2", "right_knee"),
    ("4EJ5", "right_knee"),
    ("5EJ6", "right_knee"),
    ("5EJ9", "right_knee"),
    ("6EJ5", "right_knee"),
    ("5EJ10", "right_knee"),
    ("6EJ3", "right_elbow"),
]

GT_MARKER_IDS = (1, 2, 3)


def main() -> None:
    print(f"{'video':8s} {'joint':14s} {'offset_s':>9s} {'corr':>7s}")
    print("-" * 45)
    for video, joint in VIDEOS:
        video_path = f"{BASE}/{video}/{video}.mp4"
        txt_path = f"{BASE}/{video}/{video}.TXT"

        recording = parse_maxtraq_txt(txt_path)
        gt_times = recording.times_s
        gt_angles = marker_angle_series_deg(recording, GT_MARKER_IDS)

        with MediaPipePoseEstimator() as estimator:
            result = process_video(video_path, "diag", estimator, ModelName.MEDIAPIPE)

        times_s = [f.timestamp_ms / 1000 for f in result.frames]
        angles = []
        for frame in result.frames:
            kpts = [
                Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
                for kp in frame.keypoints
            ]
            angles.append(compute_joint_angles(kpts)[joint])

        res = find_best_time_offset(gt_times, gt_angles, times_s, angles, search_range_s=(-5.0, 15.0), min_overlap_fraction=0.5)
        print(f"{video:8s} {joint:14s} {res['offset_s']:9.2f} {res['correlation']:7.3f}")


if __name__ == "__main__":
    main()
