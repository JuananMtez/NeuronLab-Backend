from typing import Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.models import models
from app.crud import csv as csv_crud
from app.crud import experiment as experiment_crud
from app.crud import subject as subject_crud
from datetime import datetime
import os
from app.schemas.csv import CSVCopy

import pandas as pd


def get_csv_by_id(db: Session, csv_id: int) -> Optional[models.CSV]:
    csv = csv_crud.find_by_id(db, csv_id)
    return csv


def get_all_csv_experiment(db: Session, experiment_id: int) -> Optional[list[models.Experiment]]:
    e = experiment_crud.find_by_id(db, experiment_id)
    if e is None:
        return None
    return e.csvs


def create_csv(db: Session, name: str, subject_id: int, experiment_id: int, file: UploadFile) -> Optional[models.CSV]:

    exp = experiment_crud.find_by_id(db, experiment_id)
    subject = subject_crud.find_by_id(db, subject_id)
    if exp is None or subject is None:
        return None

    name_file = generate_name_csv()

    csv_reader = pd.read_csv(file.file)
    csv_reader.to_csv(name_file)

    db_csv = models.CSV(name=name,
                        subject_name=subject.name + ' ' + subject.surname,
                        original=True,
                        experiment_id=experiment_id,
                        path=name_file)

    subject.total_experiments_performed = subject.total_experiments_performed + 1
    subject_crud.save(db, subject)

    return csv_crud.save(db, db_csv)


def delete_csv(db: Session, csv_id: int) -> bool:
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return False

    os.remove(csv.path)

    csv_crud.delete(db, csv)
    return True


def csv_copy(db: Session, csv_id: int, csv_copy: CSVCopy) -> Optional[models.CSV]:

    csv_original = csv_crud.find_by_id(db, csv_id)

    if csv_original is None:
        return None

    try:
        file = pd.read_csv(csv_original.path)
    except FileNotFoundError:
        return None

    name_file = generate_name_csv()
    file.to_csv(name_file)

    db_csv = models.CSV(name=csv_copy.name,
                        subject_name=csv_original.subject_name,
                        original=False,
                        experiment_id=csv_original.experiment_id,
                        path=name_file)

    return csv_crud.save(db, db_csv)


def change_name(db: Session, csv_id: int, csv_copy: CSVCopy) -> Optional[models.CSV]:

    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    csv.name = csv_copy.name
    return csv_crud.save(db, csv)


def generate_name_csv():
    now = datetime.now()
    return "csvs/record_{}.csv".format(now.strftime("%d-%m-%Y-%H-%M-%S"))