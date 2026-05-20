"""FastAPI application — serves API + static frontend."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import chat, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    from engine import Engine, get_logger

    engine = Engine.get()
    removed = engine._infra.session_store.cleanup_old_sessions(max_sessions=3)
    if removed:
        get_logger().info("AppStartup", "Startup cleanup: removed {} old session(s)".format(removed))

    yield


app = FastAPI(title="Engine Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

# Serve frontend static files (must be last — catches all routes)
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
