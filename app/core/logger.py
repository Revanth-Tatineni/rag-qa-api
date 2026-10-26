"""Logging setup for the API and the worker: human-readable locally, JSON everywhere else."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# Set per request by the request-context middleware; every log line picks it up.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

TEXT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s"


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "module": record.module,
            "line": record.lineno,
        }
        # Structured fields passed as logger.info("...", extra={"fields": {...}})
        entry.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: str, json_logs: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Our request middleware writes a richer access log than uvicorn's.
    logging.getLogger("uvicorn.access").disabled = True
    # arq installs its own handler, which would print every worker line twice.
    logging.getLogger("arq").handlers.clear()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
