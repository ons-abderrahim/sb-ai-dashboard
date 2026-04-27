from fastapi import APIRouter
router = APIRouter()

@router.get("/active")
async def get_active_anomalies():
    return {"anomalies": []}
