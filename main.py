from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "PDF Processor API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
