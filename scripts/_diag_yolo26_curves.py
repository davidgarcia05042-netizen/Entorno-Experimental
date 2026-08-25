import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.ground_truth import compare_angle_series, knee_angle_series_deg, parse_maxtraq_txt, resample_series
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.yolov8_pose import YoloV8PoseEstimator
from app.schemas.pose_result import ModelName

VIDEO = "data/gold_standard/marcha_katherine_2026-06-19/video.mp4"
MAXTRAQ = "data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT"
OFFSET = -0.02
LEG = "left_knee"
OUT = "data/gold_standard/marcha_katherine_2026-06-19/yolo26_curves.png"

recording = parse_maxtraq_txt(MAXTRAQ)
gt_times = recording.times_s
gt_angles = knee_angle_series_deg(recording)


def angle_series(estimator):
    with estimator:
        result = process_video(VIDEO, "yolo26_curves", estimator, ModelName.YOLOV8)
    times = [f.timestamp_ms / 1000 for f in result.frames]
    angles = []
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        angles.append(compute_joint_angles(kpts)[LEG])
    return times, angles


fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(gt_times, gt_angles, label="Gold standard (Maxtraq)", color="#1f2937", linewidth=2)

for label, weights, color in [("yolo26n", "yolo26n-pose.pt", "#ca8a04"), ("yolo26m", "yolo26m-pose.pt", "#dc2626")]:
    print(f"Procesando {label}...")
    times, angles = angle_series(YoloV8PoseEstimator(weights=weights))
    shifted = [t - OFFSET for t in times]
    pred_on_grid = resample_series(shifted, angles, gt_times)
    report = compare_angle_series(pred_on_grid, gt_angles)
    print(f"  -> error medio {report['mean_error_deg']}°")
    ax.plot(gt_times, pred_on_grid, label=f"{label} (err. medio {report['mean_error_deg']:.2f}°)", color=color, linewidth=1.3, alpha=0.85)

ax.set_xlabel("Tiempo (s, reloj del gold standard)")
ax.set_ylabel("Ángulo de rodilla (grados)")
ax.set_title(f"YOLO26 (nano/medium) vs. gold standard ({LEG})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"\nGráfico guardado en: {OUT}")
