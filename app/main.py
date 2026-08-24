"""Punto de entrada de la aplicación. Ejecutar con: uvicorn app.main:app --reload"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Servicio para la Evaluación de Modelos de Estimación de Pose Humana",
    description="Servicio experimental que compara MediaPipe Pose y YOLOv8 Pose "
    "bajo condiciones de oclusión parcial e iluminación variable.",
    version="0.1.0",
)

app.include_router(router)
