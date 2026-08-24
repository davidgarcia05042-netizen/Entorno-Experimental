"""
Pipeline de procesamiento de video: lee un video, aplica (opcionalmente)
simulación de iluminación y/o oclusión programada por frame, corre el
modelo de pose, y devuelve un VideoAnalysisResult listo para persistir
o comparar contra el ground truth.
"""

import time

import cv2

from app.core.illumination import IlluminationLevel, simulate_illumination, estimate_mean_brightness
from app.core.occlusion import OcclusionRegion, apply_occlusion, OcclusionMethod
from app.core.visualization import draw_frame_label, draw_pose_on_frame
from app.models.base import PoseEstimator
from app.schemas.pose_result import FrameResult, KeypointResult, ModelName, VideoAnalysisResult


def process_video(
    video_path: str,
    video_id: str,
    estimator: PoseEstimator,
    model_name: ModelName,
    illumination_level: IlluminationLevel | None = None,
    occlusion_region: OcclusionRegion | None = None,
    occlusion_method: OcclusionMethod = OcclusionMethod.BLACK_BOX,
    max_frames: int | None = None,
    output_video_path: str | None = None,
) -> VideoAnalysisResult:
    """
    Procesa un video completo. Si se pasa `illumination_level` y/o
    `occlusion_region`, esas transformaciones se aplican a CADA frame antes
    de correr el modelo, lo que permite generar las variantes experimentales
    (ej. mismo video, con oclusión programada + iluminación "dark") sin
    duplicar archivos de video en disco.

    Si se pasa `output_video_path`, además se escribe un .mp4 con el
    esqueleto detectado dibujado sobre cada frame (verde = confianza alta,
    rojo = confianza baja/inferida). El frame que se anota es el mismo que
    "vio" el modelo, es decir, ya con oclusión/iluminación simulada
    aplicada si corresponde — así el video muestra exactamente la
    condición bajo la que se evaluó.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_results: list[FrameResult] = []
    frame_index = 0
    latencies: list[float] = []

    video_writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            raise RuntimeError(
                f"No se pudo crear el video de salida en: {output_video_path}. "
                "Verifica que la carpeta exista y que el códec 'mp4v' esté disponible."
            )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        occlusion_tag = "none"

        if illumination_level is not None:
            frame = simulate_illumination(frame, illumination_level)

        if occlusion_region is not None:
            frame = apply_occlusion(frame, occlusion_region, method=occlusion_method)
            occlusion_tag = f"synthetic:{occlusion_region.x_min}-{occlusion_region.x_max}"

        start = time.perf_counter()
        keypoints = estimator.predict(frame)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        frame_results.append(
            FrameResult(
                frame_index=frame_index,
                timestamp_ms=(frame_index / fps) * 1000,
                keypoints=[
                    KeypointResult(
                        name=kp.name.value,
                        x=kp.x,
                        y=kp.y,
                        confidence=kp.confidence,
                        visible=kp.visible,
                    )
                    for kp in keypoints
                ],
                inference_latency_ms=latency_ms,
                illumination_lux_estimate=estimate_mean_brightness(frame),
                occlusion_applied=occlusion_tag,
            )
        )
        if video_writer is not None:
            annotated_frame = draw_pose_on_frame(frame, keypoints)
            annotated_frame = draw_frame_label(annotated_frame, model_name.value)
            video_writer.write(annotated_frame)

        frame_index += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    return VideoAnalysisResult(
        video_id=video_id,
        model_name=model_name,
        fps=fps,
        total_frames=frame_index,
        frames=frame_results,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )
