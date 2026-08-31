"""
Barre las variantes de tamaño/peso de cada modelo (YOLOv8 nano/small/medium,
MediaPipe Lite/Full/Heavy) contra el gold standard, reusando el offset de
sincronización ya confirmado (ver scripts/sync_video_to_ground_truth.py --
NO se recalcula por variante, mismo criterio que evaluate_against_ground_truth.py).

Uso (barrido completo):
    python -m scripts.evaluate_model_variants \
        --video data/gold_standard/marcha_katherine_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT \
        --offset -0.02 --leg left_knee \
        --out data/gold_standard/marcha_katherine_2026-06-19/variants_report.png

Uso (subconjunto de variantes + guardar keypoints/CSV de la corrida):
    python -m scripts.evaluate_model_variants \
        --video data/gold_standard/marcha_katherine_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT \
        --offset -0.02 --leg left_knee \
        --models yolo26n mediapipe_heavy \
        --out data/gold_standard/marcha_katherine_2026-06-19/variants_report.png \
        --store-dir data/gold_standard/marcha_katherine_2026-06-19

Uso (con oclusión sintética sobre la rodilla, auto-centrada -- ver
app/core/occlusion.py): igual que arriba más --occlude-knee auto. Si se
pasa --store-dir, los resultados quedan en una subcarpeta separada de los
runs sin oclusión (runs/no_occlusion/ vs. runs/occlusion_<pierna>_knee/),
para no mezclar ambas condiciones:
    python -m scripts.evaluate_model_variants \
        --video data/gold_standard/marcha_katherine_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT \
        --offset -0.02 --leg left_knee \
        --models yolo26n mediapipe_heavy mediapipe_lite \
        --occlude-knee auto \
        --out data/gold_standard/marcha_katherine_2026-06-19/variants_occluded.png \
        --store-dir data/gold_standard/marcha_katherine_2026-06-19
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.ground_truth import compare_angle_series, knee_angle_series_deg, parse_maxtraq_txt, resample_series
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.occlusion import detect_reference_knee_position, get_first_frame_shape, region_around_point
from app.core.results_store import create_run_dir, save_frame_metrics_csv, save_keypoints_json, save_run_info, save_summary_csv
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


def _angle_series(video_path: str, estimator, model_name: ModelName, leg: str, occlusion_region=None):
    with estimator:
        result = process_video(video_path, "variant_eval", estimator, model_name, occlusion_region=occlusion_region)

    times_s = [f.timestamp_ms / 1000 for f in result.frames]
    angles = []
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        angles.append(compute_joint_angles(kpts)[leg])
    return times_s, angles, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--maxtraq", required=True)
    parser.add_argument("--offset", type=float, required=True)
    parser.add_argument("--leg", required=True, choices=["left_knee", "right_knee"])
    parser.add_argument("--out", required=True, help="Gráfico de barras (error medio por variante)")
    parser.add_argument("--out-yolo-curves", help="Gráfico de líneas: variantes YOLOv8 vs. gold standard")
    parser.add_argument("--out-mediapipe-curves", help="Gráfico de líneas: variantes MediaPipe vs. gold standard")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subconjunto de variantes a correr (por defecto todas). "
        f"Disponibles: {', '.join(list(YOLO_VARIANTS) + list(MEDIAPIPE_VARIANTS))}",
    )
    parser.add_argument(
        "--store-dir",
        help="Si se da, guarda keypoints (JSON) y métricas por frame/resumen (CSV) de esta corrida "
        "en <store-dir>/runs/no_occlusion/<timestamp>/ o <store-dir>/runs/occlusion_<pierna>_knee/<timestamp>/ "
        "según --occlude-knee, para no mezclar ambas condiciones",
    )
    parser.add_argument(
        "--occlude-knee",
        choices=["auto", "fixed"],
        default=None,
        help="Aplica oclusión sintética sobre la rodilla ANTES de correr cada modelo (misma "
        "mecánica que scripts/run_single_video.py). 'auto' detecta la posición real de la "
        "rodilla en el video; 'fixed' usa una coordenada de ejemplo (40%%/60%% del frame).",
    )
    parser.add_argument(
        "--occlude-leg",
        choices=["left", "right"],
        default=None,
        help="Pierna a ocluir (default: la misma que --leg)",
    )
    parser.add_argument("--occlude-radius-px", type=int, default=60)
    args = parser.parse_args()

    yolo_variants = YOLO_VARIANTS
    mediapipe_variants = MEDIAPIPE_VARIANTS
    if args.models:
        unknown = set(args.models) - set(YOLO_VARIANTS) - set(MEDIAPIPE_VARIANTS)
        if unknown:
            raise SystemExit(f"Variantes desconocidas en --models: {sorted(unknown)}")
        yolo_variants = {k: v for k, v in YOLO_VARIANTS.items() if k in args.models}
        mediapipe_variants = {k: v for k, v in MEDIAPIPE_VARIANTS.items() if k in args.models}

    occlusion_region = None
    occlusion_leg = args.occlude_leg or args.leg.removesuffix("_knee")
    if args.occlude_knee == "auto":
        center_x, center_y = detect_reference_knee_position(args.video, occlusion_leg)
        print(f"Rodilla {occlusion_leg} detectada en ({center_x:.0f}, {center_y:.0f}) px -- oclusión centrada ahí.")
        height, width = get_first_frame_shape(args.video)
        occlusion_region = region_around_point(
            (height, width), center_x=center_x, center_y=center_y, radius_px=args.occlude_radius_px
        )
    elif args.occlude_knee == "fixed":
        height, width = get_first_frame_shape(args.video)
        occlusion_region = region_around_point(
            (height, width), center_x=width * 0.4, center_y=height * 0.6, radius_px=args.occlude_radius_px
        )

    run_dir = None
    if args.store_dir:
        subdir = f"occlusion_{occlusion_leg}_knee" if args.occlude_knee else "no_occlusion"
        run_dir = create_run_dir(args.store_dir, subdir=subdir)
    if run_dir:
        print(f"Guardando keypoints y métricas de esta corrida en: {run_dir}")

    recording = parse_maxtraq_txt(args.maxtraq)
    gt_times = recording.times_s
    gt_angles = knee_angle_series_deg(recording)

    results: dict[str, dict] = {}
    curves: dict[str, list[float]] = {}

    for label, make_estimator in yolo_variants.items():
        print(f"Procesando {label}...")
        t0 = time.perf_counter()
        try:
            times, angles, raw_result = _angle_series(
                args.video, make_estimator(), ModelName.YOLOV8, args.leg, occlusion_region=occlusion_region
            )
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
        if run_dir:
            save_keypoints_json(run_dir, label, raw_result)
        print(f"  -> error medio {report['mean_error_deg']}°  ({elapsed:.1f}s)")

    for label, complexity in mediapipe_variants.items():
        print(f"Procesando {label} (model_complexity={complexity})...")
        t0 = time.perf_counter()
        times, angles, raw_result = _angle_series(
            args.video,
            MediaPipePoseEstimator(model_complexity=complexity),
            ModelName.MEDIAPIPE,
            args.leg,
            occlusion_region=occlusion_region,
        )
        elapsed = time.perf_counter() - t0

        shifted = [t - args.offset for t in times]
        pred_on_grid = resample_series(shifted, angles, gt_times)
        report = compare_angle_series(pred_on_grid, gt_angles)
        report["elapsed_s"] = round(elapsed, 1)
        results[label] = report
        curves[label] = pred_on_grid
        if run_dir:
            save_keypoints_json(run_dir, label, raw_result)
        print(f"  -> error medio {report['mean_error_deg']}°  ({elapsed:.1f}s)")

    if run_dir:
        save_frame_metrics_csv(run_dir, gt_times, gt_angles, curves, args.leg)
        save_summary_csv(run_dir, results)
        save_run_info(
            run_dir,
            {
                "video": args.video,
                "maxtraq": args.maxtraq,
                "offset_s": args.offset,
                "leg": args.leg,
                "models": list(results.keys()),
                "occlusion_mode": args.occlude_knee,
                "occlusion_leg": occlusion_leg if args.occlude_knee else None,
                "occlusion_radius_px": args.occlude_radius_px if args.occlude_knee else None,
                "timestamp_utc": run_dir.name,
            },
        )
        print(f"Keypoints + CSV guardados en: {run_dir}")

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
