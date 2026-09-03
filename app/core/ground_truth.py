"""
Parser de exportaciones Maxtraq (captura de movimiento óptica del
laboratorio) y utilidades para comparar el ángulo de rodilla derivado de
esos marcadores contra el ángulo calculado por los modelos de pose.

Formato del archivo (confirmado con MARCHAKATHERINE19_06_2026_maxtraq_1.TXT):

    Frame number,740
    First frame,1
    Point frequency,60
    Analog frequency,60

    Time,Point #1,,,Point #2,,,...
    s,cm,cm,cm,cm,cm,cm,...
    ,X,Y,Z,X,Y,Z,...
    0,33.2903,37.8444,100.195,...
    0.0166667,...

Los "Point #N" son genéricos -- Maxtraq no exporta el nombre anatómico
del marcador, eso depende del protocolo de colocación del laboratorio.

Lo único confirmado hasta ahora (2026-08-25, comunicado por el
laboratorio): los marcadores #1, #3 y #5 son los que se usaron para el
ángulo de rodilla. Se asume el orden anatómico estándar para describir
ese ángulo -- cadera (proximal), rodilla (vértice), tobillo (distal) --
PENDIENTE de confirmación explícita del laboratorio sobre cuál marcador
es cuál punto exacto. Los marcadores #2, #4 y #6 no tienen identidad
anatómica confirmada todavía y no se usan en este módulo.

La sincronización frame-a-frame entre este archivo (60 Hz, ver
`point_frequency_hz`) y el video correspondiente tampoco está
verificada -- no se sabe si el video se grabó también a 60 fps ni si
hay un offset inicial entre el arranque de la cámara y el del mocap.
Por eso las funciones de comparación reciben las series ya alineadas
por el caller en vez de asumir una correspondencia 1:1 por índice.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import classify_angular_error

# Marcadores usados para el ángulo de rodilla, confirmados por el laboratorio.
# Orden asumido (proximal, vértice, distal) = (cadera, rodilla, tobillo);
# ver docstring del módulo -- pendiente de confirmar con el laboratorio.
KNEE_ANGLE_MARKER_IDS: tuple[int, int, int] = (1, 3, 5)

Point3D = tuple[float, float, float]


@dataclass
class MaxtraqRecording:
    """Una grabación de mocap ya parseada."""

    point_frequency_hz: float
    total_frames: int
    times_s: list[float]
    # marker_id (1-indexado, tal como aparece en el archivo) -> lista de
    # puntos por frame. None en un frame significa marcador no visible
    # (oclusión) en esa captura.
    points: dict[int, list[Point3D | None]]


def parse_maxtraq_txt(path: str | Path) -> MaxtraqRecording:
    """Parsea un archivo de exportación Maxtraq (formato .TXT, ver docstring del módulo)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    metadata: dict[str, str] = {}
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Time,"):
            header_idx = i
            break
        if "," in line:
            key, _, value = line.partition(",")
            metadata[key.strip()] = value.strip()

    if header_idx is None:
        raise ValueError(f"No se encontró la fila de encabezado 'Time,...' en {path}")

    header_cols = lines[header_idx].split(",")
    n_points = (len(header_cols) - 1) // 3

    times: list[float] = []
    points: dict[int, list[Point3D | None]] = {marker_id: [] for marker_id in range(1, n_points + 1)}

    # Después de la fila "Time,..." vienen dos filas más de encabezado
    # (unidades y ejes X/Y/Z) antes de que empiecen los datos.
    data_start = header_idx + 3
    for line in lines[data_start:]:
        if not line.strip():
            continue
        values = line.split(",")
        times.append(float(values[0]))
        for marker_id in range(1, n_points + 1):
            offset = 1 + (marker_id - 1) * 3
            x_raw, y_raw, z_raw = values[offset], values[offset + 1], values[offset + 2]
            if x_raw == "" or y_raw == "" or z_raw == "":
                points[marker_id].append(None)
            else:
                points[marker_id].append((float(x_raw), float(y_raw), float(z_raw)))

    return MaxtraqRecording(
        point_frequency_hz=float(metadata.get("Point frequency", "0")),
        total_frames=int(metadata.get("Frame number", str(len(times)))),
        times_s=times,
        points=points,
    )


def _angle_3d_deg(a: Point3D, vertex: Point3D, c: Point3D) -> float:
    """Ángulo en grados formado en `vertex` por los segmentos vertex->a y vertex->c (3D)."""
    v1 = (a[0] - vertex[0], a[1] - vertex[1], a[2] - vertex[2])
    v2 = (c[0] - vertex[0], c[1] - vertex[1], c[2] - vertex[2])

    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

    if mag1 == 0 or mag2 == 0:
        return float("nan")

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def marker_angle_series_deg(
    recording: MaxtraqRecording,
    marker_ids: tuple[int, int, int],
) -> list[float]:
    """
    Serie de ángulo por frame formado por 3 marcadores 3D (proximal,
    vértice, distal) -- pese al nombre histórico "knee" de su alias, esto
    sirve para CUALQUIER articulación de 3 marcadores (hombro, codo,
    cadera, rodilla, etc.), no solo rodilla. Qué articulación representa
    depende de `marker_ids` y de la convención de marcadores de esa
    grabación en particular (ver docstring del módulo). NaN en los frames
    donde algún marcador no fue visible.
    """
    proximal_id, vertex_id, distal_id = marker_ids
    proximal_pts = recording.points[proximal_id]
    vertex_pts = recording.points[vertex_id]
    distal_pts = recording.points[distal_id]

    angles = []
    for a, vertex, c in zip(proximal_pts, vertex_pts, distal_pts):
        if a is None or vertex is None or c is None:
            angles.append(float("nan"))
        else:
            angles.append(_angle_3d_deg(a, vertex, c))
    return angles


def knee_angle_series_deg(
    recording: MaxtraqRecording,
    marker_ids: tuple[int, int, int] = KNEE_ANGLE_MARKER_IDS,
) -> list[float]:
    """Alias retrocompatible de marker_angle_series_deg, default = marcadores de rodilla de Marcha Katherine."""
    return marker_angle_series_deg(recording, marker_ids)


def resample_series(times_s: list[float], values_deg: list[float], grid_times_s: list[float]) -> list[float]:
    """
    Interpola linealmente una serie irregular (times_s, values_deg) sobre
    los instantes de `grid_times_s`. Devuelve NaN fuera del rango cubierto
    por `times_s` o donde el valor original ya era NaN.
    """
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(values_deg, dtype=float)
    valid = ~np.isnan(values)

    grid = np.asarray(grid_times_s, dtype=float)
    if valid.sum() < 2:
        return [float("nan")] * len(grid)

    return np.interp(grid, times[valid], values[valid], left=np.nan, right=np.nan).tolist()


def find_best_time_offset(
    gt_times_s: list[float],
    gt_angles_deg: list[float],
    query_times_s: list[float],
    query_angles_deg: list[float],
    search_range_s: tuple[float, float] = (-5.0, 15.0),
    step_s: float = 0.02,
    min_overlap_fraction: float = 0.6,
) -> dict:
    """
    Sincroniza dos series de ángulo periódicas (ej. el gold standard de
    Maxtraq y el ángulo estimado por un modelo de pose sobre el video)
    buscando, por fuerza bruta sobre una grilla temporal fina, el offset
    que maximiza la correlación de Pearson entre ambas.

    Convención: si `offset_s` es el resultado, entonces
    `query_time - offset_s` corresponde al mismo instante que `gt_time`
    (es decir, hay que RESTAR `offset_s` al reloj de la serie query para
    alinearla con el reloj del gold standard).

    `min_overlap_fraction`: fracción mínima (respecto a la duración de la
    serie más corta) que debe traslaparse para que un offset sea
    candidato. Es crítico en señales periódicas como un ciclo de marcha:
    sin este mínimo, una ventana muy corta (ej. un solo ciclo) puede dar
    una correlación alta por pura casualidad de fase, no porque sea el
    desfase real. Con un traslape corto se puede "enganchar" con
    cualquier ciclo repetido de la señal, no necesariamente el correcto.

    Este método asume que ambas series avanzan al mismo ritmo relativo
    (no corrige diferencias de velocidad/escala, solo desfase). Si la
    correlación resultante es baja (< ~0.5), es señal de que el desfase
    no es el único problema (ej. fps variable del video) y no debe
    confiarse en el resultado sin revisión visual.
    """
    gt_times = np.asarray(gt_times_s, dtype=float)
    query_times = np.asarray(query_times_s, dtype=float)
    grid = np.arange(gt_times.min(), gt_times.max(), step_s)
    gt_grid = np.array(resample_series(gt_times_s, gt_angles_deg, grid.tolist()))

    shortest_duration = min(gt_times.max() - gt_times.min(), query_times.max() - query_times.min())
    min_overlap_points = int((min_overlap_fraction * shortest_duration) / step_s)

    best_offset = None
    best_corr = -2.0
    for offset in np.arange(search_range_s[0], search_range_s[1], step_s):
        shifted_query_times = [t - offset for t in query_times_s]
        query_grid = np.array(resample_series(shifted_query_times, query_angles_deg, grid.tolist()))

        mask = ~np.isnan(gt_grid) & ~np.isnan(query_grid)
        if mask.sum() < min_overlap_points:
            continue

        a, b = gt_grid[mask], query_grid[mask]
        if a.std() == 0 or b.std() == 0:
            continue

        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_offset = float(offset)

    return {
        "offset_s": best_offset,
        "correlation": best_corr,
        "min_overlap_points": min_overlap_points,
    }


def compare_angle_series(pred_angles_deg: list[float], gt_angles_deg: list[float]) -> dict:
    """
    Compara dos series de ángulos ya alineadas frame a frame (el índice i
    de ambas listas debe corresponder al mismo instante -- la alineación
    fps/offset entre video y mocap es responsabilidad del caller, ver
    docstring del módulo).
    """
    errors = [
        abs(pred - gt)
        for pred, gt in zip(pred_angles_deg, gt_angles_deg)
        if not (math.isnan(pred) or math.isnan(gt))
    ]

    if not errors:
        return {"mean_error_deg": float("nan"), "max_error_deg": float("nan"), "n_frames": 0}

    mean_error = sum(errors) / len(errors)
    return {
        "mean_error_deg": round(mean_error, 2),
        "max_error_deg": round(max(errors), 2),
        "n_frames": len(errors),
        "classification": classify_angular_error(mean_error),
    }
