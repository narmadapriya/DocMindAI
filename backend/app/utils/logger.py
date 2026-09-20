"""Backward-compatible logging facade for existing DocMindAI modules."""
from app.core.logging import configure_logging, get_logger, log_event

__all__ = ["configure_logging", "get_logger", "log_event"]
