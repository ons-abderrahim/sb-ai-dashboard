"""LangChain custom tools for the building agent."""
from langchain.tools import tool

@tool
def query_sensor_db(query: str) -> str:
    """Query the sensor database using natural language."""
    return f"[Sensor DB query result for: {query}]"

@tool
def get_occupancy_prediction(zone_id: str) -> str:
    """Get the current occupancy prediction for a zone."""
    return f"Zone {zone_id}: 82% probability of being occupied."

@tool
def get_active_anomalies(building_id: str = "all") -> str:
    """Get currently active anomaly alerts."""
    return "No active anomalies detected."

@tool
def trigger_hvac_adjustment(zone_id: str, adjustment_pct: float) -> str:
    """Trigger an HVAC adjustment for a zone (requires confirmation)."""
    return f"HVAC adjustment of {adjustment_pct}% queued for {zone_id}."

@tool
def rag_search(query: str) -> str:
    """Search CIISE lab publications and ASHRAE documents."""
    return f"[RAG search result for: {query}]"
