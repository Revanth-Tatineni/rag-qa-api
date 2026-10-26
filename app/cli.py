"""Shortcut commands, run as `python -m app.main <command>` (default: run)."""

import argparse
import subprocess
import sys
from collections.abc import Callable

from app.core.env import settings

PY = sys.executable


def _sh(*commands: list[str]) -> None:
    for command in commands:
        print("$", " ".join(command), flush=True)
        code = subprocess.call(command)
        if code != 0:
            sys.exit(code)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=not settings.is_production,
    )


def worker() -> None:
    from arq import run_worker

    from app.workers.settings import WorkerSettings

    run_worker(WorkerSettings)  # type: ignore[arg-type]


COMMANDS: dict[str, tuple[str, Callable[[], None]]] = {
    "run": ("start the API with auto-reload (local)", run),
    "worker": ("start the background job worker", worker),
    "migrate": ("apply database migrations", lambda: _sh([PY, "-m", "alembic", "upgrade", "head"])),
    "test": ("run the test suite with coverage", lambda: _sh([PY, "-m", "pytest", "--cov"])),
    "lint": (
        "ruff + format check + mypy",
        lambda: _sh(
            [PY, "-m", "ruff", "check", "."],
            [PY, "-m", "ruff", "format", "--check", "."],
            [PY, "-m", "mypy", "app", "evals"],
        ),
    ),
    "format": (
        "auto-fix lint and formatting",
        lambda: _sh([PY, "-m", "ruff", "check", "--fix", "."], [PY, "-m", "ruff", "format", "."]),
    ),
    "eval": (
        "retrieval benchmark (local models, no API cost)",
        lambda: _sh([PY, "-m", "evals.run_retrieval"]),
    ),
    "infra": (
        "start only Postgres + Redis in Docker",
        lambda: _sh(["docker", "compose", "up", "-d", "db", "redis"]),
    ),
    "build": ("build the Docker images", lambda: _sh(["docker", "compose", "build"])),
    "up": (
        "build and start the full stack in Docker",
        lambda: _sh(["docker", "compose", "up", "--build", "-d"]),
    ),
    "down": ("stop the Docker stack", lambda: _sh(["docker", "compose", "down"])),
    "logs": (
        "follow API + worker logs",
        lambda: _sh(["docker", "compose", "logs", "-f", "api", "worker"]),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="RAG QA API commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {name:<8} {help_}" for name, (help_, _) in COMMANDS.items()),
    )
    parser.add_argument("command", nargs="?", default="run", choices=COMMANDS, metavar="command")
    COMMANDS[parser.parse_args().command][1]()
