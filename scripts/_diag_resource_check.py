"""
Medición INFORMAL de recursos (RAM pico del proceso, tiempo de CPU) para
YOLO26-nano vs. MediaPipe Heavy sobre el mismo video. NO reemplaza el
benchmark riguroso de pose-benchmark/ (sin --cpu-threads 2, sin medición
de GPU por PID, corre en un entorno compartido sin aislar). Solo sirve
como referencia rápida de orden de magnitud.
"""

import os
import time

import psutil

from app.core.video_processor import process_video
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.models.yolov8_pose import YoloV8PoseEstimator
from app.schemas.pose_result import ModelName

VIDEO = "data/gold_standard/marcha_katherine_2026-06-19/video.mp4"
process = psutil.Process(os.getpid())


def measure(label, estimator, model_name):
    cpu_before = process.cpu_times()
    wall_before = time.perf_counter()
    peak_rss = process.memory_info().rss

    with estimator:
        result = process_video(VIDEO, "resource_check", estimator, model_name)
        peak_rss = max(peak_rss, process.memory_info().rss)

    wall_elapsed = time.perf_counter() - wall_before
    cpu_after = process.cpu_times()
    cpu_time = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)

    print(f"\n=== {label} ===")
    print(f"  tiempo real (wall): {wall_elapsed:.1f}s")
    print(f"  tiempo de CPU (user+sys): {cpu_time:.1f}s  (~{cpu_time/wall_elapsed*100:.0f}% de 1 nucleo equivalente)")
    print(f"  RAM (RSS) del proceso al terminar: {peak_rss / (1024**2):.0f} MB")
    print(f"  frames procesados: {result.total_frames}")


measure("YOLO26-nano", YoloV8PoseEstimator(weights="yolo26n-pose.pt"), ModelName.YOLOV8)
measure("MediaPipe Heavy", MediaPipePoseEstimator(model_complexity=2), ModelName.MEDIAPIPE)
