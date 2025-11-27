from fastapi import FastAPI
from src.routes.upload import router as upload_router
from src.routes.query import router as query_router

app = FastAPI(title="Generative AI Project - Upload & Query")

app.include_router(upload_router, prefix="/api")
app.include_router(query_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
