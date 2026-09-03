"""
Unifica, para UN modelo dentro de UN video, las 3 corridas por condición
(sin oclusión, con oclusión, con iluminación) ya guardadas por
evaluate_model_variants.py --store-dir, en una sola carpeta de comparación:

    <video_dir>/<model_label>/comparison/<timestamp_utc>/
        comparison_bar.png      barras: error medio por condición
        comparison_curve.png    curvas: gold standard + predicción de cada condición
        comparison_info.json    qué corrida (timestamp) de cada condición se usó

Toma automáticamente la corrida MÁS RECIENTE de cada condición (por
timestamp de carpeta). No vuelve a correr ningún modelo -- solo lee
metrics/summary.csv y metrics/frame_metrics.csv ya guardados.

Uso:
    python -m scripts.build_model_comparison \
        --video-dir "data/gold_standard/TOMA FRONTAL/2EJ1" \
        --model yolo26n
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ["no_occlusion", "occlusion", "illumination"]
CONDITION_LABELS = {
    "no_occlusion": "Sin oclusión",
    "occlusion": "Con oclusión",
    "illumination": "Con iluminación",
}
CONDITION_COLORS = {
    "no_occlusion": "#2563eb",
    "occlusion": "#dc2626",
    "illumination": "#d97706",
}


def _latest_run_dir(video_dir: Path, model: str, condition: str) -> Path | None:
    condition_dir = video_dir / model / condition
    if not condition_dir.is_dir():
        return None
    timestamps = sorted(p for p in condition_dir.iterdir() if p.is_dir())
    return timestamps[-1] if timestamps else None


def _read_summary(run_dir: Path, model: str) -> float:
    with (run_dir / "metrics" / "summary.csv").open(newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["variante"] == model)
    return float(row["error_medio_deg"])


def _read_frame_metrics(run_dir: Path, model: str) -> tuple[list[float], list[float], list[float]]:
    with (run_dir / "metrics" / "frame_metrics.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    gt_col = next(c for c in fieldnames if c.startswith("gt_") and c.endswith("_angle_deg"))
    times = [float(r["time_s"]) for r in rows]
    gt = [float(r[gt_col]) for r in rows]
    pred = [float(r[f"{model}_angle_deg"]) for r in rows]
    return times, gt, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True, help='Ej. "data/gold_standard/TOMA FRONTAL/2EJ1"')
    parser.add_argument("--model", required=True, help="Ej. yolo26n, mediapipe_heavy, mediapipe_lite")
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITIONS,
        help=f"Condiciones a unificar si existen (default: {' '.join(CONDITIONS)})",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    found: dict[str, Path] = {}
    for condition in args.conditions:
        run_dir = _latest_run_dir(video_dir, args.model, condition)
        if run_dir is None:
            print(f"Aviso: no hay corridas de '{condition}' para {args.model} en {video_dir} -- se omite.")
            continue
        found[condition] = run_dir

    if len(found) < 2:
        raise SystemExit(
            f"Se necesitan al menos 2 condiciones con corridas guardadas para comparar; se encontraron {len(found)}."
        )

    means = {c: _read_summary(rd, args.model) for c, rd in found.items()}
    curves = {c: _read_frame_metrics(rd, args.model) for c, rd in found.items()}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = video_dir / args.model / "comparison" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Barras: error medio por condición ---
    conditions_present = list(found.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [CONDITION_LABELS[c] for c in conditions_present],
        [means[c] for c in conditions_present],
        color=[CONDITION_COLORS[c] for c in conditions_present],
    )
    ax.axhline(5, color="green", linestyle="--", alpha=0.6, label="Aceptable (<=5°)")
    ax.axhline(10, color="orange", linestyle="--", alpha=0.6, label="Moderado (<=10°)")
    ax.set_ylabel("Error medio (grados)")
    ax.set_title(f"{args.model} -- error medio por condición")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_bar.png", dpi=120)
    plt.close(fig)

    # --- Curvas: gold standard + predicción de cada condición ---
    fig, ax = plt.subplots(figsize=(14, 5))
    times_ref, gt_ref, _ = curves[conditions_present[0]]
    ax.plot(times_ref, gt_ref, label="Gold standard (Maxtraq)", color="#1f2937", linewidth=2)
    for condition in conditions_present:
        times_c, _, pred_c = curves[condition]
        ax.plot(
            times_c, pred_c,
            label=f"{CONDITION_LABELS[condition]} (err. medio {means[condition]:.1f}°)",
            color=CONDITION_COLORS[condition], alpha=0.85, linewidth=1.3,
        )
    ax.set_xlabel("Tiempo (s, reloj del gold standard)")
    ax.set_ylabel("Ángulo (grados)")
    ax.set_title(f"{args.model} vs. gold standard -- comparación de condiciones")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_curve.png", dpi=120)
    plt.close(fig)

    info = {
        "model": args.model,
        "video_dir": str(video_dir),
        "source_runs": {c: str(rd) for c, rd in found.items()},
        "mean_error_deg": means,
        "timestamp_utc": timestamp,
    }
    (out_dir / "comparison_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Comparación guardada en: {out_dir}")
    for condition in conditions_present:
        print(f"  {CONDITION_LABELS[condition]:20s} error medio {means[condition]:.2f}° (fuente: {found[condition]})")


if __name__ == "__main__":
    main()
