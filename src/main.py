from fastapi import FastAPI
from routes import  base
from routes import  data
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings

app=FastAPI()
@app.on_event("startup")
async def startup_db_client():
    settings = get_settings()
    app.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongodb_client[settings.MONGODB_DATABASE]

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    
app.include_router(base.base_router)
app.include_router(data.data_router)

