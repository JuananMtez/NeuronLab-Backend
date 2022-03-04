from fastapi import FastAPI
from .routes.research_controller import research_controller
from .config.database import Base, engine
from .models import models;

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(research_controller)
