from fastapi import APIRouter

from app.api.assets import router as assets_router
from app.api.health import router as health_router
from app.api.index import router as index_router
from app.api.process import router as process_router
from app.api.search import router as search_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router, tags=["health"])
api_router.include_router(index_router, tags=["indexing"])
api_router.include_router(process_router, tags=["processing"])
api_router.include_router(search_router, tags=["search"])
api_router.include_router(assets_router, tags=["assets"])
