from fastapi import APIRouter, Depends, Response, File, UploadFile, Form, HTTPException
from ..config.database import get_db
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND
from sqlalchemy.orm import Session
from app.schemas.training import MachineLearningPost, TrainingResponse
from app.services import training as training_service
training_controller = APIRouter(
    prefix="/training",
    tags=["trainings"])


@training_controller.post("/machine")
async def create_training_machine(training_post: MachineLearningPost, db: Session = Depends(get_db)):
    training_service.create_training_machine(db, training_post)
    return {"ok", "ok"}

'''
@training_controller.post("/deep")
async def create_training_deep(training_post: MachineLearningPost, db: Session = Depends(get_db)):
    training_service.create_training(db, training_post)
    return {"ok", "ok"}
'''


@training_controller.delete("/{training_id}")
async def delete_training(training_id:int, db: Session = Depends(get_db)):
    training_service.delete_training(db, training_id)
    return {"ok", "ok"}


@training_controller.get("/csv/{csv_id}", response_model=list[TrainingResponse])
async def get_trainings_csv(csv_id: int, db: Session = Depends(get_db)):
    trainings = training_service.find_all_csv(db, csv_id)
    if trainings is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return trainings
