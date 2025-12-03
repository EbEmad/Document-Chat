from fastapi import FastAPI
from routes import  base
from routes import  data
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm import LLMProviderFactory

app=FastAPI()
@app.on_event("startup")
async def startup_db_client():
    settings = get_settings()
    app.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongodb_client[settings.MONGODB_DATABASE]

    llm_provider_factory=LLMProviderFactory(settings)

    # generation client
    app.generation_client=llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
    # embedding client
    app.embedding_client=llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        model_size=settings.EMBEDDING_MODEL_SIZE
    )

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    
app.include_router(base.base_router)
app.include_router(data.data_router)

