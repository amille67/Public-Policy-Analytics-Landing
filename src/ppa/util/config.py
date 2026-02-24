"""Configuration management using pydantic models and YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PPASettings(BaseModel):
    """Global settings for the PPA pipeline."""

    data_root: Path = Field(default=Path("data/raw/DATA"))
    output_root: Path = Field(default=Path("outputs"))
    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")
    figure_dpi: int = Field(default=150)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()


class ChapterConfig(BaseModel):
    """Base configuration for chapter pipelines."""

    chapter_id: str
    crs_epsg: int | None = None
    sample: int | None = None
    inputs: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_settings(default_config: Path | None = None) -> PPASettings:
    """Load global settings from YAML and env var overrides.

    Args:
        default_config: Path to default.yaml; defaults to config/default.yaml.

    Returns:
        PPASettings with env var overrides applied.
    """
    config_path = default_config or Path("config/default.yaml")
    data: dict[str, Any] = {}
    if config_path.exists():
        data = load_yaml(config_path)

    # Env var overrides
    overrides: dict[str, Any] = {}
    if "PPA_DATA_ROOT" in os.environ:
        overrides["data_root"] = os.environ["PPA_DATA_ROOT"]
    if "PPA_OUTPUT_ROOT" in os.environ:
        overrides["output_root"] = os.environ["PPA_OUTPUT_ROOT"]
    if "PPA_SEED" in os.environ:
        overrides["seed"] = int(os.environ["PPA_SEED"])
    if "PPA_LOG_LEVEL" in os.environ:
        overrides["log_level"] = os.environ["PPA_LOG_LEVEL"]

    data.update(overrides)
    return PPASettings(**data)


def load_chapter_config(chapter_yaml: Path, settings: PPASettings) -> ChapterConfig:
    """Load chapter-specific configuration.

    Args:
        chapter_yaml: Path to chXX.yaml config file.
        settings: Global settings (used to resolve paths).

    Returns:
        ChapterConfig populated from YAML.
    """
    data = load_yaml(chapter_yaml)
    return ChapterConfig(**data)
