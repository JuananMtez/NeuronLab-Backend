from typing import Optional, Any
from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.models import models
from app.crud import csv as csv_crud
from app.crud import experiment as experiment_crud
from app.crud import subject as subject_crud
from datetime import datetime
import os
from app.schemas.csv import CSVCopy, CSVFilters
import json
import numpy as np
import pandas as pd


def get_csv_by_id(db: Session, csv_id: int) -> Optional[models.CSV]:
    csv = csv_crud.find_by_id(db, csv_id)
    return csv


def get_all_csv_preproccessing(db: Session, csv_id: int) -> Optional[list[models.Preproccessing]]:
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    a = csv.preproccessing_list
    return csv.preproccessing_list


def get_all_csv_experiment(db: Session, experiment_id: int) -> Optional[list[models.Experiment]]:
    e = experiment_crud.find_by_id(db, experiment_id)
    if e is None:
        return None
    return e.csvs


def create_csv(db: Session, name: str, subject_id: int, experiment_id: int, files: list[UploadFile]) -> Optional[models.CSV]:

    exp = experiment_crud.find_by_id(db, experiment_id)
    subject = subject_crud.find_by_id(db, subject_id)
    if exp is None or subject is None:
        return None

    object = {
        "dataInput": [],
        "timestamp": [],
        "stimuli": []
    }
    name_file = generate_name_csv()

    for file in files:
        aux = json.loads(file.file.read())
        object["dataInput"].extend(aux["dataInput"])
        object["timestamp"].extend(aux["timestamp"])
        object["stimuli"].extend(aux["stimuli"])

    for stimulus in object["stimuli"]:
        cont = 0
        for label in exp.labels:
            if stimulus[0] != '0' and stimulus[0] != label.label :
                cont += 1
        if cont == len(exp.labels):
            return None

    if exp.device.type == 'eeg_headset':
        create_csv_eegheadset(object, exp, name_file)

    db_csv = models.CSV(name=name,
                        subject_name=subject.name + ' ' + subject.surname,
                        type='original',
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
                        type='copied',
                        experiment_id=csv_original.experiment_id,
                        path=name_file)

    return csv_crud.save(db, db_csv)


def change_name(db: Session, csv_id: int, csv_copy: CSVCopy) -> Optional[models.CSV]:

    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    csv.name = csv_copy.name
    return csv_crud.save(db, csv)


def apply_preproccessing(db: Session, csv_filters: CSVFilters):

    exp = None
    for csv_id in csv_filters.csvs:
        csv = csv_crud.find_by_id(db, csv_id)
        if csv is not None:
            if exp is None:
                exp = experiment_crud.find_by_id(db, csv.experiment_id)

            file = pd.read_csv(csv.path)
            name_file = generate_name_csv()
            file.to_csv(name_file)
            df = pd.read_csv(name_file)

            for prep in csv_filters.preproccessings:
                if prep.__class__.__name__ == 'CSVBandpass':
                    apply_bandpass(prep, df)
                elif prep.__class__.__name__ == 'CSVNotch':
                    apply_notch(prep, df)
                elif prep.__class__.__name__ == 'CSVDownsampling':
                    apply_downsampling(prep, df)


def generate_name_csv():
    now = datetime.now()
    return "csvs/record_{}.csv".format(now.strftime("%d-%m-%Y-%H-%M-%S"))


def create_csv_eegheadset(obj: Any, exp: models.Experiment, name_file: str):
    ch_names = []
    for x in exp.device.channels:
        ch_names.append(x.channel.name)

    obj["dataInput"] = np.concatenate(obj["dataInput"], axis=0)
    if len(ch_names) > 8:
        obj["dataInput"] = obj["dataInput"][:, :8]

    obj["timestamp"] = np.array(obj["timestamp"])
    obj["dataInput"] = np.c_[obj["timestamp"], obj["dataInput"]]
    data = pd.DataFrame(data=obj["dataInput"], columns=['Timestamp'] + ch_names)

    if len(obj["stimuli"]) != 0:
        data['Stimulus'] = 0
        for estim in obj["stimuli"]:
            ix = np.argmin(np.abs(estim[1] - obj["timestamp"]))
            data.loc[ix, 'Stimulus'] = estim[0][0]

    data.to_csv(name_file, float_format='%.3f', index=False)

def apply_bandpass(prep, csv):
    print('bandpass')

def apply_notch(prep, csv):
    print('notch')


def apply_downsampling(prep, csv):
    print('apply_downsampling')

