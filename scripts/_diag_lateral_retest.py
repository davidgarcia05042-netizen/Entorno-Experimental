"""
Diagnóstico puntual: re-evalúa con las tomas laterales (y la articulación
confirmada por el usuario) los videos que dieron correlación débil en la
primera pasada frontal: 3EJ6 (rodilla), 4EJ2 (codo, no rodilla como se
asumió antes), 4EJ5 (cadera), 6EJ5 (cadera). También re-evalúa 2EJ5 con
cadera (en vez de rodilla) sobre la toma frontal original, ya que no hay
lateral para ese video.

No es parte del pipeline principal.
"""

from app.core.ground_truth import find_best_time_offset, marker_angle_series_deg, parse_maxtraq_txt
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.schemas.pose_result import ModelName

BASE = "data/gold_standard/TOMA FRONTAL"

# (carpeta, archivo_video, joint_confirmado_por_usuario)
CASES = [
    ("3EJ6", "3EJ6 LATERAL.mp4", "right_knee"),
    ("4EJ2", "4EJ2 LATERAL.mp4", "right_elbow"),
    ("4EJ5", "4EJ5 LATERAL.mp4", "right_hip"),
    ("6EJ5", "6EJ5 LATERAL.mp4", "right_hip"),
    ("6EJ5", "6EJ5 LATERAL.mp4", "left_hip"),
    ("2EJ5", "2EJ5.mp4", "right_hip"),
]

GT_MARKER_IDS = (1, 2, 3)


def main() -> None:
    print(f"{'carpeta':8s} {'video':20s} {'joint':14s} {'offset_s':>9s} {'corr':>7s}")
    print("-" * 65)
    for folder, video_file, joint in CASES:
        video_path = f"{BASE}/{folder}/{video_file}"
        txt_path = f"{BASE}/{folder}/{folder}.TXT"

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
        print(f"{folder:8s} {video_file:20s} {joint:14s} {res['offset_s']:9.2f} {res['correlation']:7.3f}")


if __name__ == "__main__":
    main()
