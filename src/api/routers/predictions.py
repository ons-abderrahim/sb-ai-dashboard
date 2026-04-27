from fastapi import APIRouter
router = APIRouter()

@router.post("/occupancy")
async def predict_occupancy(zone_id: str, horizon_minutes: int = 15):
    return {"zone_id": zone_id, "occupied_probability": 0.82}

@router.post("/hvac-load")
async def predict_hvac(zone_id: str):
    return {"zone_id": zone_id, "predicted_load_kw": 12.4}
