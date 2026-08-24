"""
Endpoints REST del servicio experimental.

AJUSTAR ESTE ARCHIVO cuando tengan el contrato exacto de la plataforma de
telefisioterapia (rutas, nombres de campos, forma en que envían el video:
¿URL descargable, multipart upload, o referencia a un storage compartido?).
Por ahora se asume la forma más común: la plataforma envía una URL del
video (o lo sube directamente) y espera poder consultar el estado y
resultado de forma asíncrona vía polling.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.pose_result import AnalyzeVideoRequest, JobStatus, JobStatusResponse

router = APIRouter()

# Almacenamiento en memoria como placeholder. Para producción, reemplazar
# por Redis o una tabla en base de datos: este dict se pierde si el
# proceso se reinicia, y no escala a más de un worker.
_JOBS: dict[str, JobStatusResponse] = {}


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/videos/analyze", response_model=JobStatusResponse)
def analyze_video(request: AnalyzeVideoRequest, background_tasks: BackgroundTasks) -> JobStatusResponse:
    """
    Recibe una solicitud de análisis y la encola para procesamiento
    asíncrono. Devuelve inmediatamente un job_id para hacer polling.

    TODO (bloqueado): confirmar si la plataforma prefiere polling
    (GET /jobs/{id} repetido) o si necesita que este servicio le haga un
    callback/webhook cuando el resultado esté listo. Esto no se puede
    decidir sin ver su documentación de integración.
    """
    job_id = str(uuid.uuid4())
    job = JobStatusResponse(job_id=job_id, status=JobStatus.QUEUED)
    _JOBS[job_id] = job

    background_tasks.add_task(_run_analysis_job, job_id, request)

    return job


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    return job


def _run_analysis_job(job_id: str, request: AnalyzeVideoRequest) -> None:
    """
    Placeholder de la tarea real de procesamiento.

    NOTA: BackgroundTasks de FastAPI corre en el mismo proceso; sirve para
    desarrollo y para cargas moderadas, pero si el volumen de video crece,
    esto debería migrar a una cola real (Celery + Redis, o RQ) para no
    bloquear el worker de la API mientras procesa video. Se deja así por
    ahora para no introducir infraestructura adicional (Redis) antes de
    tener el resto del pipeline validado.
    """
    _JOBS[job_id].status = JobStatus.PROCESSING
    try:
        # Aquí se conecta app.core.video_processor.process_video(...) por
        # cada modelo solicitado, se guarda el resultado (disco/DB), y se
        # actualiza el estado. Se deja sin implementar hasta tener resuelto
        # el almacenamiento temporal de video (ver Anexo D.2 del documento).
        _JOBS[job_id].status = JobStatus.DONE
    except Exception as exc:  # noqa: BLE001
        _JOBS[job_id].status = JobStatus.FAILED
        _JOBS[job_id].detail = str(exc)
