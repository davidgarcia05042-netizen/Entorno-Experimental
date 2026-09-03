"""
Re-corre SOLO la condición de iluminación para los 8 videos confirmados,
tras corregir el bug de app/core/illumination.py::apply_gamma_correction
(estaba invertido: "dark" aclaraba y "bright" oscurecía). Las condiciones
no_occlusion y occlusion no se ven afectadas por ese bug y no se repiten.

Al terminar, regenera la carpeta de comparación por modelo (recoge
automáticamente la corrida de iluminación más reciente).

No es parte del pipeline reusable -- es la corrida puntual de corrección.
"""

import subprocess
import sys

BASE = "data/gold_standard/TOMA FRONTAL"
MODELS = ["yolo26n", "mediapipe_heavy", "mediapipe_lite"]
ILLUMINATION_LEVEL = "dark"

VIDEOS = [
    ("2EJ1", "left_shoulder", -0.20, (1, 2, 3)),
    ("2EJ3", "right_elbow", -0.44, (1, 2, 3)),
    ("6EJ3", "right_elbow", -0.36, (1, 2, 3)),
    ("3EJ9", "right_knee", -0.58, (1, 2, 3)),
    ("3EJ10", "right_knee", -0.16, (1, 2, 3)),
    ("5EJ6", "right_knee", -0.40, (1, 2, 3)),
    ("5EJ9", "right_knee", -0.34, (1, 2, 3)),
    ("5EJ10", "right_knee", -0.06, (1, 2, 3)),
]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    python = sys.executable
    for folder, joint, offset, gt_ids in VIDEOS:
        video_dir = f"{BASE}/{folder}"
        video_path = f"{video_dir}/{folder}.mp4"
        maxtraq_path = f"{video_dir}/{folder}.TXT"

        print(f"\n{'=' * 70}\n{folder} -- joint={joint} offset={offset}\n{'=' * 70}")

        _run([
            python, "-m", "scripts.evaluate_model_variants",
            "--video", video_path,
            "--maxtraq", maxtraq_path,
            "--offset", str(offset),
            "--joint", joint,
            "--gt-marker-ids", *[str(i) for i in gt_ids],
            "--models", *MODELS,
            "--store-dir", video_dir,
            "--illumination", ILLUMINATION_LEVEL,
        ])

        for model in MODELS:
            _run([
                python, "-m", "scripts.build_model_comparison",
                "--video-dir", video_dir,
                "--model", model,
            ])

    print(f"\n{'=' * 70}\nRe-corrida de iluminacion completa: {len(VIDEOS)} videos x {len(MODELS)} modelos.\n{'=' * 70}")


if __name__ == "__main__":
    main()
