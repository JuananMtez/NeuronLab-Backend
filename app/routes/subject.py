from fastapi import APIRouter, Depends, HTTPException, Response
from ..config.database import get_db
from sqlalchemy.orm import Session
from app.services import subject as subject_service
from starlette.status import HTTP_204_NO_CONTENT, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from app.schemas.subject import SubjectResponse, SubjectPost

subject_controller = APIRouter(
    prefix="/subject",
    tags=["subjects"])


@subject_controller.get("/", response_model=list[SubjectResponse])
async def get_all_subjects(db: Session = Depends(get_db)):
    return subject_service.get_all_subjects(db)


@subject_controller.get("/{subject_id}", response_model=SubjectResponse)
async def get_subject(subject_id: int, db: Session = Depends(get_db)):
    subject = subject_service.get_subjects(db, subject_id)
    if subject is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return subject


@subject_controller.post("/", response_model=SubjectResponse)
async def create_subject(subject: SubjectPost, db: Session = Depends(get_db)):
    s = subject_service.create_subject(db, subject)
    if s is None:
        return Response(status_code=HTTP_400_BAD_REQUEST)
    return s


@subject_controller.delete("/{subject_id}")
async def delete_subject(subject_id: int, db: Session = Depends(get_db)):
    if not subject_service.delete_subject(db, subject_id):
        return Response(status_code=HTTP_404_NOT_FOUND)
    return Response(status_code=HTTP_204_NO_CONTENT)
