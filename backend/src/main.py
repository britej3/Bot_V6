from fastapi import FastAPI
from backend.src.api import feature_flags

app = FastAPI()

app.include_router(feature_flags.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}
