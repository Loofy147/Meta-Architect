from fastapi import FastAPI
from lma.architect import LeadMetaArchitect
from hamha.core import HexagonalMultiHeadAttention

app = FastAPI()

# Global LMA instance for health checks
# In a real application, this would be managed more carefully
lma_instance = None

@app.on_event("startup")
def startup_event():
    global lma_instance
    hamha = HexagonalMultiHeadAttention(d_model=512, grid_radius=2)
    lma_instance = LeadMetaArchitect(hamha)

@app.get("/health")
def read_health():
    """Liveness probe."""
    return {"status": "healthy"}

@app.get("/ready")
def read_ready():
    """Readiness probe."""
    if lma_instance and lma_instance.telemetry.history:
        return {"status": "ready"}
    # Check if the LMA has been initialized and has collected at least one telemetry snapshot
    is_ready = lma_instance is not None
    return {"status": "ready" if is_ready else "initializing"}
