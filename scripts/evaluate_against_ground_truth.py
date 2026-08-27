"""
Evalúa uno o más modelos de pose contra un gold standard de laboratorio
(Maxtraq) ya sincronizado -- ver `scripts/sync_video_to_ground_truth.py`
para encontrar el offset y la pierna correctos ANTES de usar este script.

IMPORTANTE: el offset de sincronización es una propiedad de la GRABACIÓN
(cuándo arrancó la cámara respecto al mocap), no del modelo que se esté
evaluando. Por eso este script recibe `--offset` y `--leg` ya
confirmados como argumentos, en vez de recalcularlos por modelo -- si
cada modelo buscara su propio offset independientemente, uno con
predicciones más ruidosas puede "enganchar" con una correlación débil en
un desfase distinto (y erróneo) al real. Ver la sesión de sincronización
de MARCHAKATHERINE en la memoria del proyecto para un ejemplo concreto.

Uso:
    python -m scripts.evaluate_against_ground_truth \
        --video data/gold_standard/marcha_katherine_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT \
        --offset -0.02 --leg left_knee \
        --out data/gold_standard/marcha_katherine_2026-06-19/evaluation_report.png

Por defecto usa YOLOv8-nano y MediaPipe Full (model_complexity=1) -- para
evaluar otra variante, pasar --yolo-weights y/o --mediapipe-complexity
explícitamente. El label mostrado en consola, leyenda y título siempre
refleja la variante realmente usada, para no depender de la memoria de
quien lo corrió.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.ground_truth import compare_angle_series, knee_angle_series_deg, parse_maxtraq_txt, resample_series
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.platform_export import parse_platform_csv_angle_series
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.models.yolov8_pose import YoloV8PoseEstimator
from app.schemas.pose_result import ModelName

MODEL_KEYS = ["mediapipe", "yolov8"]
MEDIAPIPE_COMPLEXITY_LABELS = {0: "lite", 1: "full", 2: "heavy"}


def _build_models(yolo_weights: str, mediapipe_complexity: int) -> dict[str, tuple[ModelName, callable, str]]:
    """Construye (model_name, factory, label_legible) por --yolo-weights/--mediapipe-complexity."""
    yolo_label = yolo_weights.removesuffix("-pose.pt")
    mediapipe_label = f"mediapipe_{MEDIAPIPE_COMPLEXITY_LABELS[mediapipe_complexity]}"
    return {
        "mediapipe": (
            ModelName.MEDIAPIPE,
            lambda: MediaPipePoseEstimator(model_complexity=mediapipe_complexity),
            mediapipe_label,
        ),
        "yolov8": (ModelName.YOLOV8, lambda: YoloV8PoseEstimator(weights=yolo_weights), yolo_label),
    }


def _knee_angle_series(video_path: str, model_name: ModelName, make_estimator, model_key: str, leg: str) -> tuple[list[float], list[float]]:
    """Corre el modelo sobre el video y devuelve (tiempos_s, angulo_rodilla_deg)."""
    with make_estimator() as estimator:
        result = process_video(video_path, f"eval_{model_key}", estimator, model_name)

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
    parser.add_argument("--offset", type=float, required=True, help="Offset confirmado (segundos), ver sync_video_to_ground_truth.py")
    parser.add_argument("--leg", required=True, choices=["left_knee", "right_knee"])
    parser.add_argument("--models", nargs="+", default=["mediapipe", "yolov8"], choices=MODEL_KEYS)
    parser.add_argument(
        "--yolo-weights",
        default="yolov8n-pose.pt",
        help="Pesos de YOLO a usar, p.ej. yolov8n/s/m-pose.pt, yolo11n/s/m-pose.pt, yolo26n/m-pose.pt (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--mediapipe-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="model_complexity de MediaPipe: 0=lite, 1=full, 2=heavy (default: 1, full)",
    )
    parser.add_argument(
        "--platform-csv",
        help="CSV exportado por la plataforma real (MediaPipe ya corrido por ellos), columna 'Rodilla Izquierda Ángulo (°)'",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    models = _build_models(args.yolo_weights, args.mediapipe_complexity)

    print(f"Gold standard: {args.maxtraq}")
    recording = parse_maxtraq_txt(args.maxtraq)
    gt_times = recording.times_s
    gt_angles = knee_angle_series_deg(recording)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gt_times, gt_angles, label="Gold standard (Maxtraq)", color="#1f2937", linewidth=2)

    colors = {"mediapipe": "#dc2626", "yolov8": "#059669"}
    print(f"\nOffset de sincronización usado (confirmado previamente): {args.offset:+.3f}s, pierna: {args.leg}\n")
    print(f"Variantes: mediapipe={models['mediapipe'][2]}  yolov8={models['yolov8'][2]}\n")
    print(f"{'Modelo':20s} {'Error medio':>12s} {'Error max':>12s} {'Frames':>8s} {'Clasificación':>15s}")
    print("-" * 73)

    for model_key in args.models:
        model_name, make_estimator, label = models[model_key]
        print(f"Procesando video con {label}...")
        video_times, angles = _knee_angle_series(args.video, model_name, make_estimator, model_key, args.leg)

        shifted_times = [t - args.offset for t in video_times]
        pred_on_grid = resample_series(shifted_times, angles, gt_times)
        report = compare_angle_series(pred_on_grid, gt_angles)

        print(
            f"{label:20s} {report['mean_error_deg']:>10.2f}° {report['max_error_deg']:>10.2f}° "
            f"{report['n_frames']:>8d} {report['classification']:>15s}"
        )

        ax.plot(gt_times, pred_on_grid, label=f"{label} ({args.leg})", color=colors.get(model_key), alpha=0.75, linewidth=1.2)

    if args.platform_csv:
        csv_column = "Rodilla Izquierda Ángulo (°)" if args.leg == "left_knee" else "Rodilla Derecha Ángulo (°)"
        plat_times, plat_angles = parse_platform_csv_angle_series(args.platform_csv, column_name=csv_column)
        shifted_times = [t - args.offset for t in plat_times]
        pred_on_grid = resample_series(shifted_times, plat_angles, gt_times)
        report = compare_angle_series(pred_on_grid, gt_angles)

        print(
            f"{'plataforma':12s} {report['mean_error_deg']:>10.2f}° {report['max_error_deg']:>10.2f}° "
            f"{report['n_frames']:>8d} {report['classification']:>15s}"
        )
        ax.plot(gt_times, pred_on_grid, label="plataforma (MediaPipe)", color="#7c3aed", alpha=0.75, linewidth=1.2)

    variant_summary = " vs. ".join(models[k][2] for k in args.models)
    ax.set_xlabel("Tiempo (s, reloj del gold standard)")
    ax.set_ylabel("Ángulo de rodilla (grados)")
    ax.set_title(f"{variant_summary} vs. gold standard ({args.leg}, offset={args.offset:+.3f}s)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nGráfico guardado en: {args.out}")


if __name__ == "__main__":
    main()
