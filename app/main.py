"""Grok2API application entry point.

Start with:
  uv run granian --interface asgi --host 0.0.0.0 --port 8000 --workers 1 app.main:app

Multi-worker notes:
  - All workers run a lightweight account-directory sync loop (ACCOUNT_SYNC_INTERVAL env, default 30 s).
  - Only the worker that wins the advisory file lock runs the heavy AccountRefreshScheduler.
    On Linux/macOS this uses fcntl.flock; on Windows it falls back to "always be leader"
    (i.e. every worker runs the scheduler — acceptable because Windows deployments are
    almost always single-worker anyway).
"""

import asyncio
import os
import platform
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.platform.logging.logger import logger, setup_logging, reload_logging
from app.platform.config.snapshot import config as _config
from app.platform.errors import AppError
from app.platform.meta import get_project_version
from app.platform.paths import data_path
from app.platform.auth.middleware import get_internal_key
from app.platform.auth.key_registry import registry as api_key_registry
from app.providers import create_chatgpt2api_provider, create_embedded_qwen_provider
from app.providers.qwen_embed import _qwen_runtime_settings_from_config


load_dotenv()


# ---------------------------------------------------------------------------
# Scheduler leader-election via advisory file lock
# ---------------------------------------------------------------------------

_LOCK_FILE = data_path(".scheduler.lock")
_lock_fd: int | None = None


def _try_acquire_scheduler_lock() -> bool:
    """Non-blocking attempt to become the scheduler leader.

    Returns True if this worker acquired the lock (i.e. is the leader).
    Falls back to True on Windows where fcntl is unavailable.
    """
    global _lock_fd
    try:
        import fcntl
        _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(_LOCK_FILE), os.O_CREAT | os.O_WRONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd = fd
        return True
    except BlockingIOError:
        # Another worker already holds the lock.
        return False
    except (ImportError, OSError, AttributeError):
        # fcntl unavailable (Windows) or unexpected FS error — treat as leader.
        return True


def _release_scheduler_lock() -> None:
    global _lock_fd
    if _lock_fd is None:
        return
    try:
        import fcntl
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        os.close(_lock_fd)
    except (ImportError, OSError):
        pass
    _lock_fd = None

# ---------------------------------------------------------------------------
# Early logging setup (before config is loaded)
# ---------------------------------------------------------------------------

setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    file_logging=os.getenv("LOG_FILE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load configuration.
    await _config.load()
    reload_logging()
    logger.info(
        "application startup: service=any2api python={} platform={}",
        sys.version.split()[0],
        platform.system(),
    )

    # 2. Initialise account repository and bootstrap runtime table.
    from app.control.account.backends.factory import create_repository, describe_repository_target
    from app.control.account.runtime import (
        reconcile_refresh_runtime,
        set_refresh_scheduler,
        set_refresh_scheduler_leader,
        set_refresh_service,
    )
    from app.control.account.scheduler import get_account_refresh_scheduler
    from app.dataplane.account import get_account_directory

    storage_backend, storage_target = describe_repository_target()
    logger.info("account storage configured: backend={} target={}", storage_backend, storage_target)

    repo      = create_repository()
    await repo.initialize()

    # 2a. First-boot migrations (config seed / account migration from SQLite).
    from app.platform.startup import run_startup_migrations
    await run_startup_migrations(
        config_backend=_config._get_backend(),
        account_repo=repo,
    )
    # Reload config in case it was just seeded/migrated into the backend.
    await _config.load()

    directory = await get_account_directory(repo)

    # Expose repository on app.state for admin handlers.
    app.state.repository = repo
    app.state.directory  = directory

    # 3. Account directory sync loop — all workers, lightweight incremental pull.
    #    Keeps each worker's in-memory table eventually consistent with the repo.
    #
    #    Adaptive interval strategy:
    #      - After detecting changes  → re-check in ACCOUNT_SYNC_ACTIVE_INTERVAL (default 3 s)
    #        so rapid back-to-back writes (bulk import, refresh cycle) are picked up quickly.
    #      - After N idle polls       → back off toward ACCOUNT_SYNC_INTERVAL (default 30 s).
    #    scan_changes() is an indexed DB query that costs < 1 ms when nothing changed,
    #    so polling aggressively after a change is essentially free.
    _SYNC_IDLE_INTERVAL   = int(os.getenv("ACCOUNT_SYNC_INTERVAL",        "30"))
    _SYNC_ACTIVE_INTERVAL = int(os.getenv("ACCOUNT_SYNC_ACTIVE_INTERVAL",  "3"))
    _SYNC_IDLE_AFTER      = 5   # consecutive empty polls before returning to idle pace

    async def _sync_loop() -> None:
        idle_streak = 0
        while True:
            interval = _SYNC_ACTIVE_INTERVAL if idle_streak < _SYNC_IDLE_AFTER else _SYNC_IDLE_INTERVAL
            await asyncio.sleep(interval)
            try:
                changed = await directory.sync_if_changed()
                idle_streak = 0 if changed else min(idle_streak + 1, _SYNC_IDLE_AFTER)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("account directory sync error: error={}", exc)
                idle_streak = _SYNC_IDLE_AFTER  # back off on errors

    sync_task = asyncio.create_task(_sync_loop(), name="account-dir-sync")

    # 4. Account refresh scheduler — only the leader worker.
    #    Uses an advisory file lock so exactly one process runs the heavy
    #    upstream quota-fetch loop regardless of worker count.
    from app.control.account.refresh import AccountRefreshService
    refresh_enabled = _config.get_bool("account.refresh.enabled", False)
    refresh_svc = AccountRefreshService(repo)
    set_refresh_service(refresh_svc)
    app.state.refresh_service = refresh_svc

    is_leader = _try_acquire_scheduler_lock()
    scheduler = get_account_refresh_scheduler(refresh_svc)
    set_refresh_scheduler(scheduler)
    set_refresh_scheduler_leader(is_leader)
    app.state.account_refresh_scheduler = scheduler
    app.state.account_refresh_is_leader = is_leader
    strategy_name = reconcile_refresh_runtime(refresh_enabled)
    if is_leader and strategy_name == "quota":
        logger.info("scheduler leader: pid={} strategy=quota active_sync_s={} idle_sync_s={}", os.getpid(), _SYNC_ACTIVE_INTERVAL, _SYNC_IDLE_INTERVAL)
    elif is_leader:
        logger.info("scheduler leader: pid={} strategy=random (scheduler idle) active_sync_s={} idle_sync_s={}", os.getpid(), _SYNC_ACTIVE_INTERVAL, _SYNC_IDLE_INTERVAL)
    else:
        logger.info("scheduler follower: pid={} strategy={} active_sync_s={} idle_sync_s={}", os.getpid(), strategy_name, _SYNC_ACTIVE_INTERVAL, _SYNC_IDLE_INTERVAL)

    # 5. Initialise proxy directory.
    from app.control.proxy import get_proxy_directory
    await get_proxy_directory()

    # 6. Initialise embedded qwen provider.
    qwen_provider = create_embedded_qwen_provider(internal_key=get_internal_key())
    app.state.qwen_provider = qwen_provider
    if qwen_provider is not None and getattr(qwen_provider, "app", None) is not None:
        try:
            await qwen_provider.apply_runtime_settings(_qwen_runtime_settings_from_config())
            await qwen_provider.start()
            logger.info("embedded qwen provider started: root={}", qwen_provider.root)
        except Exception as exc:
            qwen_provider.error = str(exc)
            logger.exception("embedded qwen provider startup failed: error={}", exc)
    elif qwen_provider is not None:
        logger.warning("embedded qwen provider unavailable: error={}", qwen_provider.error)

    # 7. Initialise chatgpt2api provider.
    chatgpt_provider = create_chatgpt2api_provider()
    app.state.chatgpt_provider = chatgpt_provider
    try:
        await chatgpt_provider.start()
        settings = chatgpt_provider.settings_payload()
        logger.info(
            "chatgpt2api provider initialised: enabled={} configured={} base_url={}",
            settings["enabled"],
            settings["configured"],
            settings["base_url"] or "-",
        )
    except Exception as exc:
        logger.exception("chatgpt2api provider startup failed: error={}", exc)

    logger.info("application startup completed")
    yield

    # -----------
    # Shutdown
    # -----------
    logger.info("application shutdown started")
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass

    if is_leader:
        scheduler.stop()
        _release_scheduler_lock()

    qwen_provider = getattr(app.state, "qwen_provider", None)
    if qwen_provider is not None and getattr(qwen_provider, "started", False):
        try:
            await qwen_provider.stop()
        except Exception as exc:
            logger.warning("embedded qwen provider shutdown failed: error={}", exc)

    chatgpt_provider = getattr(app.state, "chatgpt_provider", None)
    if chatgpt_provider is not None:
        try:
            await chatgpt_provider.stop()
        except Exception as exc:
            logger.warning("chatgpt2api provider shutdown failed: error={}", exc)

    set_refresh_scheduler(None)
    set_refresh_scheduler_leader(False)
    set_refresh_service(None)
    await repo.close()
    logger.info("application shutdown completed")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    openapi_tags = [
        {"name": "OpenAI - Models", "description": "Model discovery endpoints."},
        {"name": "OpenAI - Chat", "description": "Chat Completions compatible endpoints."},
        {"name": "OpenAI - Responses", "description": "Responses API compatible endpoints."},
        {"name": "OpenAI - Images", "description": "Image generation and image edit endpoints."},
        {"name": "OpenAI - Videos", "description": "Video job creation and retrieval endpoints."},
        {"name": "OpenAI - Files", "description": "Locally cached media file endpoints."},
        {"name": "Anthropic - Messages", "description": "Anthropic Messages API compatible endpoints."},
        {"name": "Admin - System", "description": "Admin system status and runtime configuration endpoints."},
        {"name": "Admin - Tokens", "description": "Admin account token management endpoints."},
        {"name": "Admin - Batch", "description": "Admin batch operation endpoints."},
        {"name": "Admin - Assets", "description": "Admin online asset management endpoints."},
        {"name": "Admin - Cache", "description": "Admin local cache management endpoints."},
        {"name": "Internal - Sync", "description": "Internal service-to-service sync endpoints."},
        {"name": "WebUI - System", "description": "WebUI authentication and system endpoints."},
        {"name": "WebUI - Chat", "description": "WebUI chat API endpoints."},
        {"name": "WebUI - Voice", "description": "WebUI voice API endpoints."},
    ]
    app = FastAPI(
        title       = "Any2API",
        version     = get_project_version(),
        description = "OpenAI-compatible API gateway for web model providers",
        openapi_tags = openapi_tags,
        lifespan    = lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # Ensure config is loaded on every request.
    @app.middleware("http")
    async def _ensure_config(request: Request, call_next):
        from app.control.account.runtime import reconcile_refresh_runtime

        await _config.load()
        reconcile_refresh_runtime()
        response = await call_next(request)
        record = getattr(request.state, "api_key_record", None)
        if (
            isinstance(record, dict)
            and record.get("key")
            and response.status_code < 400
            and (request.url.path.startswith("/v1") or request.url.path.startswith("/anthropic/"))
        ):
            await api_key_registry.record_success(str(record["key"]))
        return response

    # Global exception handler — converts AppError to JSON.
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError):
        return JSONResponse(exc.to_dict(), status_code=exc.status)

    @app.exception_handler(Exception)
    async def _generic_error_handler(request: Request, exc: Exception):
        logger.exception("unhandled application exception: error={}", exc)
        return JSONResponse(
            {"error": {"message": "Internal server error", "type": "server_error"}},
            status_code=500,
        )

    # Routers.
    from app.products.web                import router as web_router
    from app.products.openai.router      import router as openai_router
    from app.products.anthropic.router   import router as anthropic_router
    from app.products.internal.router    import router as internal_router

    app.include_router(web_router)
    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.include_router(internal_router)

    # Static assets — new statics only.
    _statics_dir = Path(__file__).resolve().parent / "statics"
    if _statics_dir.is_dir():
        app.mount("/static", StaticFiles(directory=_statics_dir), name="static")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        from fastapi.responses import FileResponse as _FR
        _ico = _statics_dir / "favicon.ico"
        if _ico.exists():
            return _FR(_ico)
        return JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/health", include_in_schema=False)
    def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    logger.error(
        "direct startup is disabled: use "
        "uv run granian --interface asgi --host 0.0.0.0 --port 8000 app.main:app"
    )
    raise SystemExit(1)
