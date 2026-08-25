"""
Parser de las exportaciones (CSV/JSON) que produce la plataforma real de
telefisioterapia al correr su propio pipeline de MediaPipe sobre un video.
Permite comparar "MediaPipe corrido por la plataforma" contra el gold
standard de laboratorio, y contra nuestro propio wrapper de MediaPipe
(mismo modelo base, configuración/implementación potencialmente distinta).

Formato CSV (columnas en español, ver `platform_mediapipe.csv`):
"Tiempo (s)", y por cada punto trackeado tres columnas de posición
normalizada (X, Y, Z) más una columna "... Ángulo (°)" -- vacía para los
puntos que no son vértice de ningún ángulo configurado para ese ejercicio
(ver el modelo SkeletonPoint documentado en memoria del proyecto).

Formato JSON:
    {
      "exercise_id": int, "exercise_name": str, "duration": float,
      "frames": [
        {"t": segundos, "points": [[x, y, z, visibility], ...33],
         "angles": {"LEFT_HIP": deg, "LEFT_KNEE": deg, ...}}
      ]
    }

El eje de tiempo de ambos formatos corresponde al mismo reloj que el
video original (confirmado: mismo total de frames y misma duración que
`video.mp4` procesado localmente), así que el offset de sincronización
ya encontrado contra el gold standard (ver
`scripts/sync_video_to_ground_truth.py`) aplica directamente aquí, sin
necesidad de volver a buscarlo.
"""

import csv
import json
from pathlib import Path


def parse_platform_csv_angle_series(
    path: str | Path,
    column_name: str = "Rodilla Izquierda Ángulo (°)",
    time_column: str = "Tiempo (s)",
) -> tuple[list[float], list[float]]:
    """
    Extrae (tiempos_s, angulo_deg) de una columna de ángulo del CSV que
    exporta la plataforma. Filas con la columna vacía se omiten (la
    plataforma no calcula ángulo en todos los puntos trackeados, solo en
    los configurados como vértice de un triplete para ese ejercicio).
    """
    times: list[float] = []
    angles: list[float] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(column_name)
            if raw is None or raw.strip() == "":
                continue
            times.append(float(row[time_column]))
            angles.append(float(raw))
    return times, angles


def parse_platform_json_angle_series(
    path: str | Path,
    codename: str = "LEFT_KNEE",
) -> tuple[list[float], list[float]]:
    """Extrae (tiempos_s, angulo_deg) de la exportación JSON de la plataforma."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    times: list[float] = []
    angles: list[float] = []
    for frame in data["frames"]:
        frame_angles = frame.get("angles", {})
        if codename not in frame_angles:
            continue
        times.append(float(frame["t"]))
        angles.append(float(frame_angles[codename]))
    return times, angles
