"""FastAPI shared dependencies."""
from src.utils.config import settings

def get_settings():
    return settings
