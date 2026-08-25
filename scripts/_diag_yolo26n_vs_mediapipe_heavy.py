import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.ground_truth import compare_angle_series, knee_angle_series_deg, parse_maxtraq_txt, resample_series
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.models.yolov8_pose import YoloV8PoseEstimator
from app.schemas.pose_result import ModelName

VIDEO = "data/gold_standard/marcha_katherine_2026-06-19/video.mp4"
MAXTRAQ = "data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT"
OFFSET = -0.02
LEG = "left_knee"
OUT_DIR = "data/gold_standard/marcha_katherine_2026-06-19"

recording = parse_maxtraq_txt(MAXTRAQ)
gt_times = recording.times_s
gt_angles = knee_angle_series_deg(recording)


def angle_series(estimator, model_name):
    with estimator:
        result = process_video(VIDEO, "diag", estimator, model_name)
    times = [f.timestamp_ms / 1000 for f in result.frames]
    angles = []
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        angles.append(compute_joint_angles(kpts)[LEG])
    return times, angles


variants = [
    ("YOLO26-nano", YoloV8PoseEstimator(weights="yolo26n-pose.pt"), ModelName.YOLOV8, "#ca8a04"),
    ("MediaPipe Heavy", MediaPipePoseEstimator(model_complexity=2), ModelName.MEDIAPIPE, "#dc2626"),
]

reports = {}
curves = {}

for label, estimator, model_name, color in variants:
    print(f"Procesando {label}...")
    times, angles = angle_series(estimator, model_name)
    shifted = [t - OFFSET for t in times]
    pred_on_grid = resample_series(shifted, angles, gt_times)
    report = compare_angle_series(pred_on_grid, gt_angles)
    reports[label] = report
    curves[label] = pred_on_grid
    print(f"  -> error medio {report['mean_error_deg']}°  error max {report['max_error_deg']}°")

# --- Grafico de barras: error medio ---
fig, ax = plt.subplots(figsize=(7, 5))
labels = list(reports.keys())
means = [reports[l]["mean_error_deg"] for l in labels]
colors = [c for _, _, _, c in variants]
ax.bar(labels, means, color=colors)
ax.axhline(5, color="green", linestyle="--", alpha=0.6, label="Aceptable (<=5°)")
ax.axhline(10, color="orange", linestyle="--", alpha=0.6, label="Moderado (<=10°)")
ax.set_ylabel("Error medio (grados)")
ax.set_title(f"YOLO26-nano vs. MediaPipe Heavy ({LEG})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/yolo26n_vs_mediapipe_heavy_error.png", dpi=120)
print(f"\nGrafico de barras guardado.")

# --- Grafico de curvas ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(gt_times, gt_angles, label="Gold standard (Maxtraq)", color="#1f2937", linewidth=2)
for label, _, _, color in variants:
    ax.plot(
        gt_times,
        curves[label],
        label=f"{label} (err. medio {reports[label]['mean_error_deg']:.2f}°)",
        color=color,
        linewidth=1.3,
        alpha=0.85,
    )
ax.set_xlabel("Tiempo (s, reloj del gold standard)")
ax.set_ylabel("Ángulo de rodilla (grados)")
ax.set_title(f"YOLO26-nano vs. MediaPipe Heavy vs. gold standard ({LEG})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/yolo26n_vs_mediapipe_heavy_curves.png", dpi=120)
print("Grafico de curvas guardado.")
