"""
Orquesta la corrida completa (3 condiciones x 3 modelos) sobre los 8 videos
de TOMA FRONTAL confirmados con correlación alta (>=0.98) contra el gold
standard, tras el proceso de sincronización/identificación de marcador
documentado en las sesiones previas. Al terminar las 3 condiciones de un
video, construye la carpeta de comparación por modelo (build_model_comparison).

Videos EXCLUIDOS por ahora (correlación débil incluso con la articulación
correcta, pendiente de investigar): 3EJ7, 5EJ7 (pie derecho), 3EJ6, 4EJ2,
4EJ5, 6EJ5, 2EJ5.

No es parte del pipeline reusable -- es la corrida puntual de este batch.
Corre subprocesos de evaluate_model_variants.py / build_model_comparison.py
para reusar exactamente el mismo camino de código ya probado por CLI.
"""

import subprocess
import sys

BASE = "data/gold_standard/TOMA FRONTAL"
MODELS = ["yolo26n", "mediapipe_heavy", "mediapipe_lite"]
ILLUMINATION_LEVEL = "dark"  # peor caso documentado en app/core/illumination.py

# (carpeta, joint, offset, gt_marker_ids)
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

        print(f"\n{'=' * 70}\n{folder} -- joint={joint} offset={offset} gt_marker_ids={gt_ids}\n{'=' * 70}")

        base_cmd = [
            python, "-m", "scripts.evaluate_model_variants",
            "--video", video_path,
            "--maxtraq", maxtraq_path,
            "--offset", str(offset),
            "--joint", joint,
            "--gt-marker-ids", *[str(i) for i in gt_ids],
            "--models", *MODELS,
            "--store-dir", video_dir,
        ]

        _run(base_cmd)  # no_occlusion
        _run(base_cmd + ["--occlude-joint", joint])  # occlusion, auto-centrada en la articulación medida
        _run(base_cmd + ["--illumination", ILLUMINATION_LEVEL])  # illumination

        for model in MODELS:
            _run([
                python, "-m", "scripts.build_model_comparison",
                "--video-dir", video_dir,
                "--model", model,
            ])

    print(f"\n{'=' * 70}\nBatch completo: {len(VIDEOS)} videos x {len(MODELS)} modelos x 3 condiciones.\n{'=' * 70}")


if __name__ == "__main__":
    main()
