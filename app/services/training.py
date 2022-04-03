from typing import Optional

import pandas as pd
from pandas import DataFrame
from sqlalchemy.orm import Session

import app.crud.csv as csv_crud
import app.crud.training as training_crud
from app.models import models
from app.schemas.training import MachineLearningPost
from datetime import datetime
from joblib import dump, load
import app.crud.experiment as experiment_crud
import app.services.csv as csv_service
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, classification_report



def create_training_machine(db: Session, training_post: MachineLearningPost):
    exp = experiment_crud.find_by_id(db, training_post.exp_id)
    if exp is None:
        return None

    clf = None
    description = ''
    db_training = models.Training(
        name=training_post.name,
        experiment_id=training_post.exp_id,
        training_data=str(training_post.training_data) + "%"
    )

    dfs = []
    for x in training_post.csvs:
        c = csv_crud.find_by_id(db, x)
        if c is not None:
            try:
                df = pd.read_csv(c.path)
                dfs.append(df)
                db_training.csvs.append(c)
                if db_training.features is None:
                    text = ""
                    for x in c.feature_extractions:
                        text = text + str(x.feature_extraction) + ", "
                    text = text[:-2]
                    db_training.features = text


            except FileNotFoundError:
                pass

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(n=int((training_post.training_data * df.shape[0])/100))

    rawdata = df.values

    labels = np.array([df['Stimulus'].tolist()]).T

    if training_post.algorithm.__class__.__name__ == 'KNN':
        clf = apply_knn(training_post.algorithm.n_neighbors, rawdata, labels)
        description = "KNN (n_neighbors: " + str(training_post.algorithm.n_neighbors) + ")"

    elif training_post.algorithm.__class__.__name__ == 'RandomForest':
        clf = apply_random_forest(training_post.algorithm.max_depth, training_post.algorithm.n_estimators, training_post.algorithm.random_state, rawdata, labels)
        description = "Random Forest (max_depth: " + str(training_post.algorithm.max_depth) + ", n_estimatos: " + str(training_post.algorithm.n_estimators) + ", random_state: " + str(training_post.algorithm.random_state) + ")"

    elif training_post.algorithm.__class__.__name__ == 'SVM':
        clf = apply_svm(training_post.algorithm.kernel, rawdata, labels)
        description = "SVM (kernel: " + training_post.algorithm.kernel + ")"

    name_model = generate_name_model()

    db_training.path = name_model
    db_training.description = description

    dump(clf, db_training.path)

    training_crud.save(db, db_training)


def delete_training(db: Session, training_id: int):
    training = training_crud.find_by_id(db, training_id)
    training_crud.delete(db, training)


def find_all_csv(db: Session, csv_id: int) -> Optional[list[models.Training]]:
    csv = csv_crud.find_by_id(db, csv_id)
    if csv is None:
        return None
    return csv.trainings


def generate_name_model():
    now = datetime.now()
    return "models/record_{}.joblib".format(now.strftime("%d-%m-%Y-%H-%M-%S"))

def apply_knn(n_neighbors: int, rawdata, labels):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X=rawdata, y=labels)
    return clf

def apply_random_forest(max_depth: int, n_estimators: int, random_state: int, rawdata, labels):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf.fit(X=rawdata, y=labels)
    return clf

def apply_svm(kernel: str, rawdata, labels):
    clf = SVC(kernel=kernel)
    clf.fit(X=rawdata, y=labels)
    return clf


def find_all_predictable(db: Session, csv_id: int) -> Optional[list[models.Training]]:
    csv = csv_crud.find_by_id(db, csv_id)
    if csv is None:
        return None

    exp = experiment_crud.find_by_id(db, csv.experiment_id)

    if exp is None:
        return None

    text = ""
    for feature in csv.feature_extractions:
        text = text + str(feature.feature_extraction) + ", "
    text = text[:-2]

    trainings = []
    for train in exp.trainings:
        if train.features == text:
            found = False
            i = 0
            while i < len(train.csvs) and not found:
                if train.csvs[i].id == csv_id:
                    found = True
                i = i + 1
            if not found:
                trainings.append(train)

    return trainings

def predict(db: Session, training_id: int, csv_id: int):

    training = training_crud.find_by_id(db, training_id)
    if training is None:
        return
    csv = csv_crud.find_by_id(db, csv_id)
    if csv is None:
        return

    df = pd.read_csv(csv.path)
    rawdata = df.values
    labels = np.array([df['Stimulus'].tolist()]).T

    clf = load(training.path)
    cont = 0
    try:
        text = str(classification_report(labels, clf.predict(rawdata)))
    except ValueError as e:
        return str(e)

    for x in text:
        if x == '\n':
            cont += 1

    return {"text": text, "n_jumps": cont}
