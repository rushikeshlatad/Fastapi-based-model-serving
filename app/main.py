import logging
import time

from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
    status
)

from fastapi.security import APIKeyHeader

from app.config import settings

from app.logging_config import (
    configure_logging
)

from app.model import ModelManager

from app.schemas import (
    PredictionRequest,
    PredictionResponse
)


# ============================================================
# Logging
# ============================================================

configure_logging()

logger = logging.getLogger(__name__)


# ============================================================
# Model Manager
# ============================================================

model_manager = ModelManager(
    model_path=settings.model_path,
    metadata_path=settings.metadata_path
)


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting %s",
        settings.app_name
    )

    logger.info(
        "Loading model..."
    )

    try:

        model_manager.load_model()

        logger.info(
            "Model loaded successfully"
        )

        logger.info(
            "Model version: %s",
            settings.model_version
        )

    except Exception:

        logger.exception(
            "Model loading failed"
        )

        raise

    yield

    logger.info(
        "Application shutting down"
    )


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-style Random Forest "
        "classification API"
    ),
    lifespan=lifespan
)


# ============================================================
# API Key Security
# ============================================================

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str | None = Security(
        api_key_header
    )
):

    if api_key != settings.api_key:

        logger.warning(
            "Unauthorized request"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key


# ============================================================
# Request Logging Middleware
# ============================================================

@app.middleware("http")
async def logging_middleware(
    request: Request,
    call_next
):

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "%s %s -> %s | %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    response.headers[
        "X-Process-Time"
    ] = f"{duration:.6f}"

    return response


# ============================================================
# Root
# ============================================================

@app.get("/")
async def root():

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }


# ============================================================
# Liveness
# ============================================================

@app.get(
    "/health/live"
)
async def liveness():

    return {
        "status": "alive"
    }


# ============================================================
# Readiness
# ============================================================

@app.get(
    "/health/ready"
)
async def readiness():

    if not model_manager.is_ready():

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Model is not ready"
        )

    return {
        "status": "ready",
        "model_version": (
            settings.model_version
        )
    }


# ============================================================
# Model Information
# ============================================================

@app.get(
    "/api/v1/model-info"
)
async def model_info(
    _: str = Depends(verify_api_key)
):

    metadata = (
        model_manager.get_metadata()
    )

    return {
        "model_version": (
            settings.model_version
        ),
        "model_type": metadata.get(
            "model_type"
        ),
        "dataset": metadata.get(
            "dataset"
        ),
        "number_of_features": metadata.get(
            "n_features"
        ),
        "target_classes": metadata.get(
            "target_names"
        )
    }


# ============================================================
# Prediction
# ============================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    dependencies=[
        Depends(verify_api_key)
    ]
)
async def predict(
    request: PredictionRequest
):

    try:

        prediction, probabilities = (
            model_manager.predict(
                request.features
            )
        )

        predicted_class = int(
            prediction[0]
        )

        probability = float(
            probabilities[
                predicted_class
            ]
        )

        metadata = (
            model_manager.get_metadata()
        )

        target_names = metadata.get(
            "target_names",
            ["class_0", "class_1"]
        )

        class_name = target_names[
            predicted_class
        ]

        return PredictionResponse(
            prediction=predicted_class,
            class_name=class_name,
            probability=probability,
            model_version=(
                settings.model_version
            )
        )

    except Exception:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Prediction failed"
        )