"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.chat import router as chat_router
from backend.api.routes.control_center import router as control_center_router
from backend.config.settings import Settings, get_settings
from backend.core.errors import ConfigurationError, register_exception_handlers
from backend.core.logging import configure_logging, log_event
from backend.llm.engine import LLMEngineManager
from backend.llm.service import ChatCompletionService
from backend.repointel.service import RepositoryIntelligenceEngine
from backend.runtime.api import MultiAgentRuntime

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

def create_app(*, start_runtime: bool = True) -> FastAPI:
    engine_manager = LLMEngineManager(settings)
    chat_service = ChatCompletionService(settings, engine_manager)
    repo_intel = RepositoryIntelligenceEngine(settings)
    multi_agent_runtime = MultiAgentRuntime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_runtime:
            await engine_manager.initialize()
            try:
                repo_status = await repo_intel.verify_runtime()
                log_event(
                    logger,
                    logging.INFO,
                    "repo.startup.ready",
                    "Repository intelligence runtime checks passed.",
                    **repo_status,
                )
            except Exception as exc:
                raise ConfigurationError(
                    f"Repository intelligence startup verification failed: {exc}"
                ) from exc
            multi_agent_runtime.register_default_mock_agents()
            await multi_agent_runtime.start()
        try:
            yield
        finally:
            await multi_agent_runtime.stop()
            await repo_intel.shutdown()
            if start_runtime:
                await engine_manager.shutdown()

    app = FastAPI(title="Forge Gateway", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(control_center_router)
    app.state.settings = settings
    app.state.engine_manager = engine_manager
    app.state.chat_service = chat_service
    app.state.repo_intel = repo_intel
    app.state.multi_agent_runtime = multi_agent_runtime

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
