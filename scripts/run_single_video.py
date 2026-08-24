"""
Script de prueba rápida: corre ambos modelos sobre un video local y guarda
los resultados en JSON. Útil para validar el pipeline ANTES de conectar la
API REST, y para generar los primeros resultados exploratorios del Sprint 2.

Uso:
    python scripts/run_single_video.py --video ruta/al/video.mp4 --out resultados/

Ejemplo con oclusión programada + iluminación oscura simulada:
    python scripts/run_single_video.py --video ruta/al/video.mp4 \
        --out resultados/ --occlude-knee --illumination dark
"""

import argparse
import json
from pathlib import Path

import cv2

from app.core.illumination import IlluminationLevel
from app.core.occlusion import region_around_point
from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.models.yolov8_pose import YoloV8PoseEstimator
from app.schemas.pose_result import ModelName


def _get_first_frame_shape(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"No se pudo leer el video: {video_path}")
    return frame.shape[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Ruta al video local (.mp4)")
    parser.add_argument("--out", required=True, help="Carpeta de salida para los JSON de resultados")
    parser.add_argument(
        "--illumination",
        choices=[level.value for level in IlluminationLevel],
        default=None,
        help="Simula un nivel de iluminación con OpenCV (ver app/core/illumination.py)",
    )
    parser.add_argument(
        "--occlude-knee",
        action="store_true",
        help="Ejemplo de oclusión programada: cubre la zona de la rodilla izquierda "
        "(coordenada aproximada fija, ajustar a su set de pruebas real)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Además del JSON, guarda un .mp4 con el esqueleto dibujado sobre cada frame "
        "(verde = confianza alta, rojo = confianza baja/inferida)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    illumination_level = IlluminationLevel(args.illumination) if args.illumination else None

    occlusion_region = None
    if args.occlude_knee:
        height, width = _get_first_frame_shape(args.video)
        # Coordenada de ejemplo: ajustar una vez tengan keypoints reales de
        # referencia de ese video específico, o pasar por un primer pase
        # de detección para ubicar la rodilla automáticamente.
        occlusion_region = region_around_point(
            (height, width), center_x=width * 0.4, center_y=height * 0.6, radius_px=60
        )

    video_id = Path(args.video).stem

    models = {
        ModelName.MEDIAPIPE: MediaPipePoseEstimator(),
        ModelName.YOLOV8: YoloV8PoseEstimator(),
    }

    for model_name, estimator in models.items():
        print(f"Procesando con {model_name.value}...")

        output_video_path = None
        if args.save_video:
            output_video_path = str(out_dir / f"{video_id}_{model_name.value}_annotated.mp4")

        with estimator:
            result = process_video(
                video_path=args.video,
                video_id=video_id,
                estimator=estimator,
                model_name=model_name,
                illumination_level=illumination_level,
                occlusion_region=occlusion_region,
                max_frames=args.max_frames,
                output_video_path=output_video_path,
            )

        out_path = out_dir / f"{video_id}_{model_name.value}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"  -> {out_path} ({result.total_frames} frames, "
              f"latencia media {result.mean_latency_ms:.1f} ms)")
        if output_video_path:
            print(f"  -> {output_video_path}")


if __name__ == "__main__":
    main()
