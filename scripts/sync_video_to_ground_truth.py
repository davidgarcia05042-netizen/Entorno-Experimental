"""
Sincroniza un video con su archivo Maxtraq correspondiente: corre un modelo
de pose sobre el video, extrae su serie de ángulo de rodilla (izq. y der.),
y busca por correlación cruzada el desfase temporal que mejor la alinea
contra el gold standard -- en vez de pedirle a una persona que identifique
a ojo el frame exacto de máxima flexión (poco confiable en videos de
caminadora, donde el ciclo de marcha es rápido y repetitivo).

Uso:
    python -m scripts.sync_video_to_ground_truth \
        --video data/gold_standard/marcha_katherine_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_katherine_2026-06-19/maxtraq.TXT \
        --out data/gold_standard/marcha_katherine_2026-06-19/sync_report.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.ground_truth import (
    find_best_time_offset,
    knee_angle_series_deg,
    parse_maxtraq_txt,
    resample_series,
)
from app.core.metrics import compute_joint_angles
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.schemas.pose_result import ModelName


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--maxtraq", required=True)
    parser.add_argument("--out", required=True, help="Ruta del gráfico de diagnóstico (.png)")
    args = parser.parse_args()

    print(f"Procesando video con MediaPipe: {args.video}")
    with MediaPipePoseEstimator() as estimator:
        result = process_video(
            video_path=args.video,
            video_id="sync_check",
            estimator=estimator,
            model_name=ModelName.MEDIAPIPE,
        )

    video_times_s = [f.timestamp_ms / 1000 for f in result.frames]

    # compute_joint_angles espera objetos Keypoint reales; reconstruirlos
    # desde KeypointResult (el schema usado para persistir resultados).
    from app.core.keypoint_schema import Keypoint, UnifiedKeypoint

    left_angles, right_angles = [], []
    for frame in result.frames:
        kpts = [
            Keypoint(
                name=UnifiedKeypoint(kp.name),
                x=kp.x,
                y=kp.y,
                confidence=kp.confidence,
                visible=kp.visible,
            )
            for kp in frame.keypoints
        ]
        angles = compute_joint_angles(kpts)
        left_angles.append(angles["left_knee"])
        right_angles.append(angles["right_knee"])

    print(f"Parseando gold standard: {args.maxtraq}")
    recording = parse_maxtraq_txt(args.maxtraq)
    gt_times_s = recording.times_s
    gt_angles = knee_angle_series_deg(recording)

    print("\nBuscando mejor desfase (correlación cruzada, exigiendo traslape >= 60% del video)...")
    results = {}
    for label, query_angles in [("left_knee", left_angles), ("right_knee", right_angles)]:
        r = find_best_time_offset(gt_times_s, gt_angles, video_times_s, query_angles)
        results[label] = r
        if r["offset_s"] is None:
            print(f"  {label:12s}: sin candidatos con traslape suficiente ({r['min_overlap_points']} puntos mínimo)")
        else:
            print(f"  {label:12s}: offset={r['offset_s']:+.3f}s  correlacion={r['correlation']:.3f}")

    best_label = max(results, key=lambda k: results[k]["correlation"])
    best = results[best_label]
    if best["offset_s"] is None:
        print("\nNingún candidato tuvo traslape suficiente en el rango de búsqueda. No se puede sincronizar así.")
        return
    print(f"\n=> Mejor coincidencia: {best_label} (correlacion={best['correlation']:.3f}, offset={best['offset_s']:+.3f}s)")
    if best["correlation"] < 0.5:
        print("   ADVERTENCIA: correlación baja. No confiar en este offset sin revisión visual --")
        print("   puede haber más de un desfase (ej. fps variable del video) además del offset simple.")

    # Gráfico de diagnóstico: gold standard vs. la pierna ganadora, ya alineada.
    best_query_angles = left_angles if best_label == "left_knee" else right_angles
    shifted_times = [t - best["offset_s"] for t in video_times_s]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gt_times_s, gt_angles, label="Gold standard (Maxtraq)", color="#2563eb", linewidth=1.5)
    ax.plot(shifted_times, best_query_angles, label=f"Video ({best_label}, MediaPipe, alineado)", color="#dc2626", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Tiempo (s, reloj del gold standard)")
    ax.set_ylabel("Ángulo de rodilla (grados)")
    ax.set_title(f"Sincronización video<->gold standard  (offset={best['offset_s']:+.3f}s, r={best['correlation']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nGráfico de diagnóstico guardado en: {args.out}")
    print("Revisa visualmente que las dos curvas se superpongan en fase antes de confiar en el offset.")


if __name__ == "__main__":
    main()
