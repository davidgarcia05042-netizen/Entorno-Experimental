"""
Resumen de métricas que SÍ se pueden calcular sin ground truth:
  - Latencia (media, mediana, máxima)
  - Tasa de detección general
  - Tasa de detección separada: frames con oclusión vs. sin oclusión

Uso:
    python -m scripts.show_metrics --results resultados/mi_video_mediapipe_pose.json
"""

import argparse
import json
import statistics
from pathlib import Path

from app.config import DETECTION_CONFIDENCE_THRESHOLD
from app.core.keypoint_schema import UnifiedKeypoint, Keypoint


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


def _detection_rate(keypoints: list[Keypoint]) -> float:
    if not keypoints:
        return float("nan")
    detected = sum(1 for kp in keypoints if kp.confidence >= DETECTION_CONFIDENCE_THRESHOLD)
    return detected / len(keypoints)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Ruta al JSON generado por run_single_video.py")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))

    frames = data["frames"]
    latencies = [f["inference_latency_ms"] for f in frames]

    occluded_rates = []
    visible_rates = []

    for frame in frames:
        keypoints = _keypoints_from_frame(frame)
        rate = _detection_rate(keypoints)
        if frame.get("occlusion_applied") not in (None, "none"):
            occluded_rates.append(rate)
        else:
            visible_rates.append(rate)

    print(f"=== {data['model_name']} — {data['video_id']} ===")
    print(f"Total de frames: {data['total_frames']}")
    print()
    print("--- Latencia (ms) ---")
    print(f"  Media:   {statistics.mean(latencies):.2f}")
    print(f"  Mediana: {statistics.median(latencies):.2f}")
    print(f"  Máxima:  {max(latencies):.2f}")
    print()
    print("--- Tasa de detección (confianza >= "
          f"{DETECTION_CONFIDENCE_THRESHOLD}) ---")
    if visible_rates:
        print(f"  Frames sin oclusión ({len(visible_rates)}): {statistics.mean(visible_rates):.1%}")
    if occluded_rates:
        print(f"  Frames con oclusión ({len(occluded_rates)}): {statistics.mean(occluded_rates):.1%}")
    if visible_rates and occluded_rates:
        delta = statistics.mean(visible_rates) - statistics.mean(occluded_rates)
        print(f"  Degradación por oclusión: {delta:.1%}")
    if not occluded_rates:
        print("  (Este video no tiene frames marcados con oclusión programada — "
              "corre con --occlude-knee para generar esa comparación)")


if __name__ == "__main__":
    main()
