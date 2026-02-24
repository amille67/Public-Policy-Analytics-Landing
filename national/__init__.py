"""National-scale orchestration and source loaders."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "NationalOrchestrator":
        from .orchestrator import NationalOrchestrator

        return NationalOrchestrator
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return ["NationalOrchestrator"]

__all__ = ["NationalOrchestrator"]
