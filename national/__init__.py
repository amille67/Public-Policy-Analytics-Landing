"""National-scale orchestration and source loaders."""


def __getattr__(name: str):
    if name == "NationalOrchestrator":
        from .orchestrator import NationalOrchestrator

        return NationalOrchestrator
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return ["NationalOrchestrator"]

__all__ = ["NationalOrchestrator"]
