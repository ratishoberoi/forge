"""Application-specific exceptions and handlers."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse


class ForgeError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_type: str = "forge_error",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = dict(details or {})


class ConfigurationError(ForgeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500, error_type="configuration_error")


class ModelNotReadyError(ForgeError):
    def __init__(self, message: str = "Model runtime is not ready.") -> None:
        super().__init__(message, status_code=503, error_type="model_not_ready")


class InvalidRequestError(ForgeError):
    def __init__(self, message: str, *, details: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            status_code=400,
            error_type="invalid_request_error",
            details=details,
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ForgeError)
    async def handle_forge_error(_: Request, exc: ForgeError) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Unhandled server error.",
                    "type": "internal_server_error",
                    "details": {"exception": exc.__class__.__name__},
                }
            },
        )
