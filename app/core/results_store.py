"""Persistencia de resultados por ejecución contra el gold standard.

Cada ejecución guarda, bajo una carpeta con timestamp (para no sobrescribir
corridas anteriores): los keypoints crudos por modelo (JSON, mismo esquema
que `run_single_video.py`), el ángulo/error por frame de cada modelo frente
al gold standard (CSV), el resumen por variante (CSV) y los metadatos de la
corrida (offset de sincronización, pierna, video, timestamp).

Pensado para vivir bajo `data/gold_standard/<video>/`, que ya está en
`.gitignore` -- estos datos se derivan de video de paciente y no deben
publicarse a git.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.pose_result import VideoAnalysisResult


def create_run_dir(base_dir: str | Path) -> Path:
    """Crea (y devuelve) `<base_dir>/runs/<timestamp_utc>/`."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_keypoints_json(run_dir: Path, model_label: str, result: VideoAnalysisResult) -> Path:
    """Guarda los keypoints crudos por frame de un modelo (mismo esquema que run_single_video.py)."""
    out_path = Path(run_dir) / f"{model_label}_keypoints.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def save_frame_metrics_csv(
    run_dir: Path,
    gt_times_s: list[float],
    gt_angles_deg: list[float],
    model_curves: dict[str, list[float]],
    leg: str,
) -> Path:
    """Guarda ángulo predicho + error absoluto por frame de cada modelo, sobre la grilla del gold standard."""
    out_path = Path(run_dir) / "frame_metrics.csv"
    model_labels = list(model_curves.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["frame_idx", "time_s", f"gt_{leg}_angle_deg"]
        for label in model_labels:
            header += [f"{label}_angle_deg", f"{label}_error_deg"]
        writer.writerow(header)
        for i, (t, gt) in enumerate(zip(gt_times_s, gt_angles_deg)):
            row = [i, round(t, 4), round(gt, 3)]
            for label in model_labels:
                pred = model_curves[label][i]
                row += [round(pred, 3), round(abs(pred - gt), 3)]
            writer.writerow(row)
    return out_path


def save_summary_csv(run_dir: Path, reports: dict[str, dict]) -> Path:
    """Guarda el resumen por variante (error medio/máximo, clasificación, tiempo)."""
    out_path = Path(run_dir) / "summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variante", "error_medio_deg", "error_max_deg", "n_frames", "clasificacion", "tiempo_s"])
        for label, report in reports.items():
            writer.writerow(
                [
                    label,
                    report["mean_error_deg"],
                    report["max_error_deg"],
                    report["n_frames"],
                    report["classification"],
                    report.get("elapsed_s", ""),
                ]
            )
    return out_path


def save_run_info(run_dir: Path, info: dict) -> Path:
    """Guarda metadatos de la corrida (video, maxtraq, offset, pierna, modelos, timestamp)."""
    out_path = Path(run_dir) / "run_info.json"
    out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
