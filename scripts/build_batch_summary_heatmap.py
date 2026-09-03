"""
Genera un mapa de calor (heatmap) con el error medio (grados) de los 8
videos de TOMA FRONTAL confirmados x 3 modelos x 3 condiciones, leyendo
comparison_info.json de cada video/modelo (la corrida más reciente de
cada condición). Pensado como gráfico único para documentación (tesis),
resumiendo la tabla general en una sola imagen.

Uso:
    python -m scripts.build_batch_summary_heatmap
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

BASE = Path("data/gold_standard/TOMA FRONTAL")
VIDEOS = ["2EJ1", "2EJ3", "6EJ3", "3EJ9", "3EJ10", "5EJ6", "5EJ9", "5EJ10"]
MODELS = ["yolo26n", "mediapipe_heavy", "mediapipe_lite"]
CONDITIONS = ["no_occlusion", "occlusion", "illumination"]
CONDITION_LABELS = ["Sin oclusión", "Con oclusión", "Iluminación (dark)"]

OUT_PATH = BASE / "batch_summary_heatmap.png"


def main() -> None:
    row_labels = []
    data = []
    for video in VIDEOS:
        for model in MODELS:
            comp_dir = BASE / video / model / "comparison"
            if not comp_dir.is_dir():
                continue
            latest = sorted(comp_dir.iterdir())[-1]
            info = json.loads((latest / "comparison_info.json").read_text(encoding="utf-8"))
            m = info["mean_error_deg"]
            row_labels.append(f"{video} · {model}")
            data.append([m.get(c, float("nan")) for c in CONDITIONS])

    matrix = np.array(data)

    bounds = [0, 5, 10, 20, 100]
    colors = ["#16a34a", "#eab308", "#f97316", "#dc2626"]  # verde, amarillo, naranja, rojo
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    fig_height = 0.42 * len(row_labels) + 2
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITION_LABELS)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(len(row_labels)):
        for j in range(len(CONDITIONS)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.1f}°", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    # separador visual entre videos (cada 3 filas = 1 video x 3 modelos)
    for i in range(3, len(row_labels), 3):
        ax.axhline(i - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, ticks=[2.5, 7.5, 15, 30], fraction=0.05, pad=0.03)
    cbar.ax.set_yticklabels(["≤5° aceptable", "5-10° moderado", "10-20°", ">20° no aceptable"], fontsize=7)

    ax.set_title("Error medio (grados) por video x modelo x condición -- TOMA FRONTAL", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Heatmap guardado en: {OUT_PATH}")


if __name__ == "__main__":
    main()
