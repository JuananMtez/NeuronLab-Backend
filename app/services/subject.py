from sqlalchemy.orm import Session
from app.schemas.subject import SubjectResponse, SubjectPost
from app.models import models
from app.crud import subject as subject_crud
from typing import Optional


def get_all_subjects(db:Session) -> list[SubjectResponse]:
    return subject_crud.find_all(db)


def get_subjects(db:Session, subject_id: int) -> Optional[SubjectResponse]:
    return subject_crud.find_by_id(db, subject_id)



def create_subject(db: Session, subject: SubjectPost) -> models.Subject:

    db_subject = models.Subject(name=subject.name,
                                surname=subject.surname,
                                age=subject.age,
                                total_experiments_performed=0)

    for x in subject.mental_conditions:
        condition = models.MentalCondition(condition=x.condition)
        db_subject.mental_conditions.append(condition)

    return subject_crud.save(db, db_subject)


def delete_subject(db: Session, subject_id: int) -> bool:
    subject = subject_crud.find_by_id(db, subject_id)
    if subject is None:
        return False;

    subject_crud.delete(db, subject)
    return True