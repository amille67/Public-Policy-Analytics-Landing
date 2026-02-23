"""Custom exceptions for PPA pipelines."""


class PPAError(Exception):
    """Base exception for all PPA pipeline errors."""


class MissingColumnError(PPAError):
    """Raised when a required column is missing from a DataFrame."""

    def __init__(self, column: str, source: str) -> None:
        super().__init__(f"Required column '{column}' not found in {source}")


class InvalidCRSError(PPAError):
    """Raised when a GeoDataFrame has an invalid or geographic CRS."""


class DataValidationError(PPAError):
    """Raised when data fails a validation check."""
