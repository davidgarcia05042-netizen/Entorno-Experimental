"""
Diagnóstico puntual: ¿el error mayor de YOLOv8 se debe a inestabilidad
(jitter frame-a-frame, baja confianza en los puntos usados para el
ángulo) más que a una mala localización promedio? Compara jitter y
confianza entre YOLOv8 (nano y medium) y MediaPipe (full y heavy) sobre
el mismo video.
"""

import numpy as np

from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.models.yolov8_pose import YoloV8PoseEstimator
from app.schemas.pose_result import ModelName

VIDEO = "data/gold_standard/marcha_katherine_2026-06-19/video.mp4"
LEG = "left_knee"
POINTS = {
    "left_knee": (UnifiedKeypoint.LEFT_HIP, UnifiedKeypoint.LEFT_KNEE, UnifiedKeypoint.LEFT_ANKLE),
}[LEG]


def analyze(label, estimator, model_name):
    with estimator:
        result = process_video(VIDEO, f"diag_{label}", estimator, model_name)

    angles = []
    conf_by_point = {p.value: [] for p in POINTS}
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        angles.append(compute_joint_angles(kpts)[LEG])
        by_name = {kp.name: kp for kp in kpts}
        for p in POINTS:
            conf_by_point[p.value].append(by_name[p].confidence)

    angles = np.array(angles, dtype=float)
    jitter = np.abs(np.diff(angles))
    jitter = jitter[~np.isnan(jitter)]

    print(f"\n=== {label} ===")
    print(f"  jitter medio frame-a-frame: {jitter.mean():.2f}°   jitter max: {jitter.max():.2f}°")
    print(f"  frames con salto > 15° respecto al frame anterior: {(jitter > 15).sum()} / {len(jitter)}")
    for point_name, confs in conf_by_point.items():
        confs = np.array(confs)
        print(
            f"  confianza {point_name:18s}: media={confs.mean():.3f}  "
            f"min={confs.min():.3f}  frac<0.5={float((confs < 0.5).mean()):.2%}"
        )


analyze("yolov8n", YoloV8PoseEstimator(weights="yolov8n-pose.pt"), ModelName.YOLOV8)
analyze("yolov8m", YoloV8PoseEstimator(weights="yolov8m-pose.pt"), ModelName.YOLOV8)
analyze("mediapipe_full", MediaPipePoseEstimator(model_complexity=1), ModelName.MEDIAPIPE)
analyze("mediapipe_heavy", MediaPipePoseEstimator(model_complexity=2), ModelName.MEDIAPIPE)
