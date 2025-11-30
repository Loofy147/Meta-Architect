from fastapi import FastAPI
from lma.architect import LeadMetaArchitect
from hamha.core import HexagonalMultiHeadAttention

app = FastAPI()

# Global LMA instance for health checks
# In a real application, this would be managed more carefully
lma_instance = None


@app.on_event("startup")
def startup_event():
    """Initializes the global LMA instance on application startup.

    This function creates a default `HexagonalMultiHeadAttention` model and
    a corresponding `LeadMetaArchitect` instance to be used by the API
    endpoints. In a production environment, this setup would be more
    configurable.
    """
    global lma_instance
    hamha = HexagonalMultiHeadAttention(d_model=512, grid_radius=2)
    lma_instance = LeadMetaArchitect(hamha)


@app.get("/health")
def read_health():
    """Provides a basic liveness probe endpoint.

    This endpoint can be used by container orchestration systems (like
    Kubernetes) to check if the application is running.

    Returns:
        dict: A dictionary with a "status" key indicating that the service
            is healthy.
    """
    return {"status": "healthy"}


@app.get("/ready")
def read_ready():
    """Provides a readiness probe endpoint.

    This endpoint checks if the `LeadMetaArchitect` has been initialized. In a
    more complete implementation, it would also check for the availability of
    downstream services or other dependencies.

    Returns:
        dict: A dictionary with a "status" key indicating whether the service
            is "ready" or "initializing".
    """
    if lma_instance and lma_instance.telemetry.history:
        return {"status": "ready"}
    # Check if the LMA has been initialized and has collected at least one telemetry snapshot
    is_ready = lma_instance is not None
    return {"status": "ready" if is_ready else "initializing"}
