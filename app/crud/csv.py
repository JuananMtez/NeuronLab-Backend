from sqlalchemy.orm import Session
from app.models import models


def find_by_id(db: Session, csv_id: int) -> models.CSV:
    return db.query(models.CSV).filter(models.CSV.id == csv_id).first()


def save(db: Session, csv: models.CSV) -> models.CSV:
    db.add(csv)
    db.commit()
    db.refresh(csv)
    return csv


def find_all(db: Session) -> list[models.CSV]:
    return db.query(models.CSV).all()


def delete(db: Session, csv: models.CSV):
    db.delete(csv)
    db.commit()
