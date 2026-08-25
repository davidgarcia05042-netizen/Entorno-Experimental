"""
Barre las variantes de tamaño/peso de cada modelo (YOLOv8 nano/small/medium,
MediaPipe Lite/Full/Heavy) contra el gold standard, reusando el offset de
sincronización ya confirmado (ver scripts/sync_video_to_ground_truth.py --
NO se recalcula por variante, mismo criterio que evaluate_against_ground_truth.py).

Uso:
    python -m scripts.evaluate_model_variants \
        --video data/gold_standard/marcha_katherine_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT \
        --offset -0.02 --leg left_knee \
        --out data/gold_standard/marcha_katherine_2026-06-19/variants_report.png
"""

import argparse
import time

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

YOLO_VARIANTS = {
    "yolov8n": lambda: YoloV8PoseEstimator(weights="yolov8n-pose.pt"),
    "yolov8s": lambda: YoloV8PoseEstimator(weights="yolov8s-pose.pt"),
    "yolov8m": lambda: YoloV8PoseEstimator(weights="yolov8m-pose.pt"),
    "yolo11n": lambda: YoloV8PoseEstimator(weights="yolo11n-pose.pt"),
    "yolo11s": lambda: YoloV8PoseEstimator(weights="yolo11s-pose.pt"),
    "yolo11m": lambda: YoloV8PoseEstimator(weights="yolo11m-pose.pt"),
    # conf/iou explícitos en la inferencia (no solo post-filtrado), ver
    # el hallazgo del repo de referencia dcortesav/pose-estimation-YOLO26
    "yolov8n_hiconf": lambda: YoloV8PoseEstimator(weights="yolov8n-pose.pt", inference_conf=0.80, inference_iou=0.45),
    "yolov8m_hiconf": lambda: YoloV8PoseEstimator(weights="yolov8m-pose.pt", inference_conf=0.80, inference_iou=0.45),
    "yolo26n": lambda: YoloV8PoseEstimator(weights="yolo26n-pose.pt"),
    "yolo26m": lambda: YoloV8PoseEstimator(weights="yolo26m-pose.pt"),
}

MEDIAPIPE_VARIANTS = {
    "mediapipe_lite": 0,
    "mediapipe_full": 1,
    "mediapipe_heavy": 2,
}


def _angle_series(video_path: str, estimator, model_name: ModelName, leg: str) -> tuple[list[float], list[float]]:
    with estimator:
        result = process_video(video_path, "variant_eval", estimator, model_name)

    times_s = [f.timestamp_ms / 1000 for f in result.frames]
    angles = []
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        angles.append(compute_joint_angles(kpts)[leg])
    return times_s, angles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--maxtraq", required=True)
    parser.add_argument("--offset", type=float, required=True)
    parser.add_argument("--leg", required=True, choices=["left_knee", "right_knee"])
    parser.add_argument("--out", required=True, help="Gráfico de barras (error medio por variante)")
    parser.add_argument("--out-yolo-curves", help="Gráfico de líneas: variantes YOLOv8 vs. gold standard")
    parser.add_argument("--out-mediapipe-curves", help="Gráfico de líneas: variantes MediaPipe vs. gold standard")
    args = parser.parse_args()

    recording = parse_maxtraq_txt(args.maxtraq)
    gt_times = recording.times_s
    gt_angles = knee_angle_series_deg(recording)

    results: dict[str, dict] = {}
    curves: dict[str, list[float]] = {}

    for label, make_estimator in YOLO_VARIANTS.items():
        print(f"Procesando {label}...")
        t0 = time.perf_counter()
        try:
            times, angles = _angle_series(args.video, make_estimator(), ModelName.YOLOV8, args.leg)
        except Exception as exc:
            print(f"  ERROR con {label}: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        shifted = [t - args.offset for t in times]
        pred_on_grid = resample_series(shifted, angles, gt_times)
        report = compare_angle_series(pred_on_grid, gt_angles)
        report["elapsed_s"] = round(elapsed, 1)
        results[label] = report
        curves[label] = pred_on_grid
        print(f"  -> error medio {report['mean_error_deg']}°  ({elapsed:.1f}s)")

    for label, complexity in MEDIAPIPE_VARIANTS.items():
        print(f"Procesando {label} (model_complexity={complexity})...")
        t0 = time.perf_counter()
        times, angles = _angle_series(
            args.video, MediaPipePoseEstimator(model_complexity=complexity), ModelName.MEDIAPIPE, args.leg
        )
        elapsed = time.perf_counter() - t0

        shifted = [t - args.offset for t in times]
        pred_on_grid = resample_series(shifted, angles, gt_times)
        report = compare_angle_series(pred_on_grid, gt_angles)
        report["elapsed_s"] = round(elapsed, 1)
        results[label] = report
        curves[label] = pred_on_grid
        print(f"  -> error medio {report['mean_error_deg']}°  ({elapsed:.1f}s)")

    print()
    print(f"{'Variante':16s} {'Error medio':>12s} {'Error max':>10s} {'Tiempo':>8s} {'Clasificación':>15s}")
    print("-" * 70)
    for label, report in results.items():
        print(
            f"{label:16s} {report['mean_error_deg']:>10.2f}° {report['max_error_deg']:>9.2f}° "
            f"{report['elapsed_s']:>7.1f}s {report['classification']:>15s}"
        )

    labels = list(results.keys())
    means = [results[label]["mean_error_deg"] for label in labels]
    colors = ["#2563eb" if label.startswith("yolo") else "#dc2626" for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, means, color=colors)
    ax.axhline(5, color="green", linestyle="--", alpha=0.6, label="Aceptable (<=5°)")
    ax.axhline(10, color="orange", linestyle="--", alpha=0.6, label="Moderado (<=10°)")
    ax.set_ylabel("Error medio (grados)")
    ax.set_title(f"Error medio por variante vs. gold standard ({args.leg})")
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nGráfico de barras guardado en: {args.out}")

    def _plot_family_curves(family_labels: list[str], out_path: str, title: str, palette: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(gt_times, gt_angles, label="Gold standard (Maxtraq)", color="#1f2937", linewidth=2)
        for label, color in zip(family_labels, palette):
            if label not in curves:
                continue
            ax.plot(gt_times, curves[label], label=f"{label} (err. medio {results[label]['mean_error_deg']:.1f}°)", color=color, alpha=0.8, linewidth=1.2)
        ax.set_xlabel("Tiempo (s, reloj del gold standard)")
        ax.set_ylabel("Ángulo de rodilla (grados)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        print(f"Gráfico de curvas guardado en: {out_path}")

    if args.out_yolo_curves:
        n_yolo = len(YOLO_VARIANTS)
        yolo_palette = plt.cm.viridis([i / max(n_yolo - 1, 1) for i in range(n_yolo)])
        _plot_family_curves(
            list(YOLO_VARIANTS.keys()),
            args.out_yolo_curves,
            f"YOLOv8 / YOLO11 / YOLO26 (variantes) vs. gold standard ({args.leg})",
            yolo_palette,
        )

    if args.out_mediapipe_curves:
        _plot_family_curves(
            list(MEDIAPIPE_VARIANTS.keys()),
            args.out_mediapipe_curves,
            f"MediaPipe (variantes) vs. gold standard ({args.leg})",
            ["#fca5a5", "#ef4444", "#b91c1c"],
        )


if __name__ == "__main__":
    main()
