from fastapi import APIRouter
router = APIRouter()

@router.get("/{zone_id}/latest")
async def get_latest(zone_id: str):
    return {"zone_id": zone_id}

@router.get("/{zone_id}/history")
async def get_history(zone_id: str, hours: int = 24):
    return {"zone_id": zone_id, "hours": hours}
