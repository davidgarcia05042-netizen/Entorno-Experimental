"""Resume, leyendo comparison_info.json de cada video/modelo, el error medio por condición. No es parte del pipeline principal."""

import json
from pathlib import Path

BASE = Path("data/gold_standard/TOMA FRONTAL")
VIDEOS = ["2EJ1", "2EJ3", "6EJ3", "3EJ9", "3EJ10", "5EJ6", "5EJ9", "5EJ10"]
MODELS = ["yolo26n", "mediapipe_heavy", "mediapipe_lite"]


def main() -> None:
    print(f"{'video':7s} {'modelo':16s} {'sin_oclusion':>13s} {'con_oclusion':>13s} {'iluminacion':>12s}")
    print("-" * 65)
    for video in VIDEOS:
        for model in MODELS:
            comp_dir = BASE / video / model / "comparison"
            if not comp_dir.is_dir():
                continue
            latest = sorted(comp_dir.iterdir())[-1]
            info = json.loads((latest / "comparison_info.json").read_text(encoding="utf-8"))
            m = info["mean_error_deg"]
            print(
                f"{video:7s} {model:16s} "
                f"{m.get('no_occlusion', float('nan')):13.2f} "
                f"{m.get('occlusion', float('nan')):13.2f} "
                f"{m.get('illumination', float('nan')):12.2f}"
            )


if __name__ == "__main__":
    main()
