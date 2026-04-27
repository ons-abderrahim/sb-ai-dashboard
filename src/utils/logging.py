import logging
import structlog

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level))
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )
