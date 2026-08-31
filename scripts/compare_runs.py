"""
Compara dos corridas YA GUARDADAS por evaluate_model_variants.py --store-dir
(ej. runs/no_occlusion/<timestamp> vs. runs/occlusion_left_knee/<timestamp>
del mismo video) sin volver a correr los modelos -- lee directamente
summary.csv y frame_metrics.csv de cada corrida.

Genera:
  --out-bar         gráfico de barras agrupado (error medio por modelo,
                     condición A vs. condición B)
  --out-curves-dir   una gráfica de curvas por modelo (gold standard +
                     predicción en cada condición superpuestas)

Uso:
    python -m scripts.compare_runs \
        --run-a data/gold_standard/marcha_katherine_2026-06-19/runs/no_occlusion/20260831T230911Z \
        --label-a "Sin oclusión" \
        --run-b data/gold_standard/marcha_katherine_2026-06-19/runs/occlusion_left_knee/20260831T231038Z \
        --label-b "Con oclusión (rodilla izq.)" \
        --out-bar data/gold_standard/marcha_katherine_2026-06-19/compare_occlusion_bar.png \
        --out-curves-dir data/gold_standard/marcha_katherine_2026-06-19/compare_occlusion_curves
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_summary(run_dir: Path) -> dict[str, float]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        return {row["variante"]: float(row["error_medio_deg"]) for row in csv.DictReader(f)}


def _read_frame_metrics(run_dir: Path) -> tuple[list[float], list[float], dict[str, list[float]]]:
    with (run_dir / "frame_metrics.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    gt_col = next(c for c in fieldnames if c.startswith("gt_") and c.endswith("_angle_deg"))
    model_labels = [
        c.removesuffix("_angle_deg") for c in fieldnames if c.endswith("_angle_deg") and not c.startswith("gt_")
    ]
    times = [float(r["time_s"]) for r in rows]
    gt = [float(r[gt_col]) for r in rows]
    curves = {label: [float(r[f"{label}_angle_deg"]) for r in rows] for label in model_labels}
    return times, gt, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="Carpeta de la corrida A (ej. runs/no_occlusion/<ts>)")
    parser.add_argument("--label-a", required=True, help="Nombre legible de la condición A")
    parser.add_argument("--run-b", required=True, help="Carpeta de la corrida B (ej. runs/occlusion_left_knee/<ts>)")
    parser.add_argument("--label-b", required=True, help="Nombre legible de la condición B")
    parser.add_argument("--out-bar", required=True)
    parser.add_argument("--out-curves-dir", required=True)
    args = parser.parse_args()

    run_a, run_b = Path(args.run_a), Path(args.run_b)
    summary_a = _read_summary(run_a)
    summary_b = _read_summary(run_b)

    labels = [label for label in summary_a if label in summary_b]
    if not labels:
        raise SystemExit("Las dos corridas no comparten ninguna variante en común (summary.csv).")
    missing_a = set(summary_b) - set(summary_a)
    missing_b = set(summary_a) - set(summary_b)
    if missing_a or missing_b:
        print(f"Aviso: variantes ignoradas por no estar en ambas corridas -- solo en A: {missing_b}, solo en B: {missing_a}")

    # --- Gráfico de barras agrupado ---
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [summary_a[label] for label in labels], width, label=args.label_a, color="#2563eb")
    ax.bar(x + width / 2, [summary_b[label] for label in labels], width, label=args.label_b, color="#dc2626")
    ax.axhline(5, color="green", linestyle="--", alpha=0.6, label="Aceptable (<=5°)")
    ax.axhline(10, color="orange", linestyle="--", alpha=0.6, label="Moderado (<=10°)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error medio (grados)")
    ax.set_title(f"Error medio por modelo -- {args.label_a} vs. {args.label_b}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_bar, dpi=120)
    print(f"Gráfico de barras guardado en: {args.out_bar}")

    # --- Curvas por modelo ---
    times_a, gt_a, curves_a = _read_frame_metrics(run_a)
    _, _, curves_b = _read_frame_metrics(run_b)

    out_dir = Path(args.out_curves_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(times_a, gt_a, label="Gold standard (Maxtraq)", color="#1f2937", linewidth=2)
        ax.plot(
            times_a, curves_a[label], label=f"{args.label_a} (err. medio {summary_a[label]:.2f}°)",
            color="#2563eb", alpha=0.85, linewidth=1.3,
        )
        ax.plot(
            times_a, curves_b[label], label=f"{args.label_b} (err. medio {summary_b[label]:.2f}°)",
            color="#dc2626", alpha=0.85, linewidth=1.3,
        )
        ax.set_xlabel("Tiempo (s, reloj del gold standard)")
        ax.set_ylabel("Ángulo de rodilla (grados)")
        ax.set_title(f"{label}: {args.label_a} vs. {args.label_b}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f"{label}_compare_curves.png"
        plt.savefig(out_path, dpi=120)
        print(f"Gráfico de curvas guardado en: {out_path}")


if __name__ == "__main__":
    main()
