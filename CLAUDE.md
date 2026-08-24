# Contexto del proyecto — Servicio de Evaluación de Modelos de Estimación de Pose Humana

## Qué es esto

Trabajo de grado (Universidad Antonio Nariño, Ing. de Sistemas) de Juan David
García Rubio y Camila Sandoval Ruiz, dirigido por Ph.D. David Alberto Herrera
Álvarez, en el marco del grupo de investigación LACSER/Bioingeniería.

**Objetivo del trabajo de grado**: desarrollar un servicio que compare
MediaPipe Pose vs. YOLOv8-Pose para estimación de pose humana, evaluando su
precisión bajo oclusión parcial e iluminación variable, con miras a
integrarse a una plataforma de telefisioterapia existente. Metodología SCRUM,
4 sprints, 12 semanas.

## Los dos subproyectos

- **`pose-evaluation-service/`**: el servicio en sí. Wrappers de MediaPipe y
  YOLOv8, esquema unificado de keypoints, métricas de precisión (MPJPE, error
  angular, PCK, tasa de detección), simulación de oclusión e iluminación,
  esqueleto de API REST, y scripts para procesar video y ver resultados.
- **`pose-benchmark/`**: entorno SEPARADO, deliberadamente mínimo, solo para
  medir tiempo/CPU/RAM/GPU de cada variante de cada modelo. Separado a
  propósito para que el overhead de FastAPI no contamine las mediciones de
  desempeño puro.

## Decisiones técnicas clave (y por qué)

- **Esquema unificado de keypoints**: MediaPipe (33 landmarks) y YOLOv8-Pose
  (17, formato COCO) no comparten esquema nativo. Se usa COCO-17 como
  vocabulario común internamente. IMPORTANTE: la plataforma de
  telefisioterapia real usa los 33 índices de MediaPipe como su convención
  (`LEFT_SHOULDER`=11, etc., en mayúsculas) — cualquier salida final hacia la
  plataforma real debe hablar en esos términos, no en COCO-17.
- **Umbral de error angular**: ≤5° aceptable, 5-10° moderado, >10° no
  aceptable. Fundamentado en Gajdosik & Bohannon (1987, *Physical Therapy*)
  y corroborado con revisiones sistemáticas recientes de mocap markerless
  vs. marker-based (ver `app/config.py` para las constantes).
- **Oclusión**: diseño mixto — oclusión NATURAL ya presente en las tomas del
  laboratorio + oclusión SINTÉTICA programada (máscaras sobre región,
  `app/core/occlusion.py`).
- **Iluminación**: los videos del laboratorio se grabaron con luz constante.
  Plan B (ya implementado): simulación con OpenCV (gamma/brillo) en
  `app/core/illumination.py`. Puede que el laboratorio grabe una segunda
  tanda con luz real variable — no confirmado aún.
- **Contrato con la plataforma real de telefisioterapia**: es un patrón de
  **webhook**, no polling. La plataforma hace `POST /video/` (multipart:
  video + points + exercise + `resultsEndpoint`), espera 200 inmediato, y el
  servicio debe llamar de vuelta a `resultsEndpoint` cuando termine, con
  `{error, results: {points}, max_angle, min_angle}`. Repo de referencia:
  `santivarelaagent-cmd/telerehabilitacion_be` (documentado vía DeepWiki).
- **Hardware de despliegue real**: Render, 2 CPU, 8GB RAM, **sin GPU**. Por
  eso YOLOv8-nano es la variante principal recomendada, no las más pesadas.

## Bloqueadores activos (no inventar soluciones para esto, preguntar)

1. **Formato de exportación de Maxtraq/Kinovea/Moca** (ground truth de
   laboratorio): desconocido. Sin esto no se puede escribir el parser
   (`app/core/ground_truth.py`, no existe todavía). Bloquea la comparación
   real contra ground truth.
2. **Mecanismo de disparo del flujo `Exercise`** en la plataforma: según su
   propia documentación, no está implementado del lado de la plataforma
   (sí existe para el flujo `ExerciseResult`).
3. **Autenticación entre los dos servicios**: no definida (RNF-SEG pendiente).

## Estado del benchmarking (pose-benchmark/)

- YOLOv8 n/s/m/l/x medidos en CPU sin límite de hilos → **no representativo
  de Render** (usaba ~700% CPU, Render solo da 2 núcleos ≈ 200% techo).
- Se corrigió: `--cpu-threads 2` en `benchmark_yolov8.py` para simular el
  límite real. También se corrigió la medición de GPU para que sea por PID
  del proceso (NVML vía `nvidia-ml-py`), no de toda la tarjeta (antes se
  contaminaba con otras apps del sistema).
- Pendiente: correr con `--cpu-threads 2` y con `--device cuda` (datos
  limpios) para tener el set completo. Fase 2 (benchmark MediaPipe:
  Lite/Full/Heavy, `model_complexity` 0/1/2) todavía no se ha construido.

## Convenciones del código

- Identificadores en inglés, docstrings/comentarios largos en español
  (el equipo sustenta en español).
- Entorno de desarrollo: Windows, PowerShell, VS Code, Python 3.12, venv por
  proyecto (`venv\Scripts\Activate.ps1`).
- Scripts se corren como módulo: `python -m scripts.nombre_script`, no
  `python scripts\nombre_script.py` (por resolución del paquete `app`).
- Todo cambio de código se verifica con `pytest tests/test_core.py -v`
  antes de entregarse (9 tests en `pose-evaluation-service`).
