"""
Compara dos resultados frame a frame usando las mismas funciones que se
usarán contra el ground truth (MPJPE, error angular, PCK). Por ahora sirve
para comparar MediaPipe vs. YOLOv8 entre sí -- útil como resultado
exploratorio de acuerdo/desacuerdo entre modelos mientras se resuelve el
parser del ground truth de laboratorio (Fase 5, bloqueada).

IMPORTANTE: esto NO reemplaza la comparación contra el ground truth real.
Que dos modelos concuerden no significa que ambos estén midiendo bien --
podrían estar los dos igual de equivocados. Es un dato exploratorio, no
una validación clínica.

Uso:
    python -m scripts.compare_results \
        --a resultados/mi_video_mediapipe_pose.json \
        --b resultados/mi_video_yolov8_pose.json
"""

import argparse
import json
import statistics
from pathlib import Path

from app.core.keypoint_schema import UnifiedKeypoint, Keypoint
from app.core.metrics import angular_error_report, mpjpe, pck


def _keypoints_from_frame(frame: dict) -> list[Keypoint]:
    return [
        Keypoint(
            name=UnifiedKeypoint(kp["name"]),
            x=kp["x"],
            y=kp["y"],
            confidence=kp["confidence"],
            visible=kp["visible"],
        )
        for kp in frame["keypoints"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="JSON del primer modelo (ej. mediapipe)")
    parser.add_argument("--b", required=True, help="JSON del segundo modelo (ej. yolov8)")
    args = parser.parse_args()

    data_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(args.b).read_text(encoding="utf-8"))

    frames_a = {f["frame_index"]: f for f in data_a["frames"]}
    frames_b = {f["frame_index"]: f for f in data_b["frames"]}
    common_indices = sorted(set(frames_a) & set(frames_b))

    if not common_indices:
        print("No hay frames en común entre los dos archivos (¿son del mismo video?)")
        return

    mpjpe_values = []
    pck_values = []
    angular_errors_by_joint: dict[str, list[float]] = {}

    for idx in common_indices:
        kp_a = _keypoints_from_frame(frames_a[idx])
        kp_b = _keypoints_from_frame(frames_b[idx])

        mpjpe_values.append(mpjpe(kp_a, kp_b, only_visible_gt=True))
        pck_values.append(pck(kp_a, kp_b))

        report = angular_error_report(kp_a, kp_b)
        for joint, result in report.items():
            if result["error_deg"] is not None:
                angular_errors_by_joint.setdefault(joint, []).append(result["error_deg"])

    valid_mpjpe = [v for v in mpjpe_values if v == v]  # descarta NaN
    valid_pck = [v for v in pck_values if v == v]

    print(f"=== Comparación: {data_a['model_name']} vs. {data_b['model_name']} ===")
    print(f"Frames comparados: {len(common_indices)}")
    print()
    print("--- Distancia entre keypoints (MPJPE, en píxeles) ---")
    if valid_mpjpe:
        print(f"  Media: {statistics.mean(valid_mpjpe):.2f} px")
        print(f"  Máxima: {max(valid_mpjpe):.2f} px")
    print()
    print("--- PCK (fracción de keypoints que coinciden dentro del umbral) ---")
    if valid_pck:
        print(f"  Media: {statistics.mean(valid_pck):.1%}")
    print()
    print("--- Error angular promedio por articulación (grados) ---")
    for joint, errors in sorted(angular_errors_by_joint.items()):
        print(f"  {joint:20s}: {statistics.mean(errors):5.2f}°  (n={len(errors)})")
    print()
    print("Recordatorio: esta comparación es entre los dos modelos, NO contra")
    print("el ground truth de laboratorio. Sirve para ver qué tanto concuerdan,")
    print("no para validar cuál de los dos está midiendo correctamente.")


if __name__ == "__main__":
    main()
