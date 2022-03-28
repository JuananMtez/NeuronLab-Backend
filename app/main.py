from fastapi import FastAPI
from app.routes import researcher, experiment, subject, csv, training
from .config.database import Base, engine
from .models import models
from fastapi.middleware.cors import CORSMiddleware


models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(researcher.researcher_controller)
app.include_router(experiment.experiment_controller)
app.include_router(subject.subject_controller)
app.include_router(csv.csv_controller)
app.include_router(training.training_controller)

