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


def create_training_machine(db: Session, training_post: MachineLearningPost):

    clf = None
    description = ''

    db_training = models.Training(
        name=training_post.name,
    )

    exp = None
    df = None
    for x in training_post.csvs:
        csv = csv_crud.find_by_id(db, x)

        if csv is None:
            break
        if exp is None:
            exp = csv.experiment_id
            db_training.experiment_id = exp
            text = ""
            for prep in csv.preproccessing_list:
                text = text + prep.description + ", "
            text = text[:-1]
            text = text[:-1]
            db_training.preproccesing_description = text

        if df is None:
            try:
                df = pd.read_csv(csv.path)
                db_training.csvs.append(csv)

            except FileNotFoundError:
                pass
        else:
            try:
                aux = pd.read_csv(csv.path)
                df = pd.concat([df, aux])
                db_training.csvs.append(csv)

            except FileNotFoundError:
                pass


    if training_post.algorithm.__class__.__name__ == 'KNN':
        clf = apply_knn(training_post.training_data, training_post.testing_data, training_post.algorithm.n_neighbors, df)
        description = "Algorithm: KNN, n_neighbors: " + str(training_post.algorithm.n_neighbors)

    elif training_post.algorithm.__class__.__name__ == 'RandomForest':
        clf = apply_random_forest(training_post.training_data, training_post.testing_data, training_post.algorithm.max_depth, training_post.algorithm.n_estimators, training_post.algorithm.random_state, df)
        description = "Algorithm: Random Forest, max_depth: " + str(training_post.algorithm.max_depth) + ", n_estimatos: " + str(training_post.algorithm.n_estimators) + ", random_state: " + str(training_post.algorithm.random_state)

    elif training_post.algorithm.__class__.__name__ == 'SVM':
        clf = apply_svm(training_post.training_data, training_post.testing_data, training_post.algorithm.kernel, df)
        description = "Algorithm: SVM, kernel: " + training_post.algorithm.kernel

    name_model = generate_name_model()
    description = description + ", training_data: " + str(training_post.training_data) + "%, testing_data: " + str(training_post.testing_data) + "%"

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

def apply_knn(training_data:int, testing_data: int, n_neighbors: int, df: DataFrame):
    pass

def apply_random_forest(training_data:int, testing_data: int, max_depth: int, n_estimators: int, random_state: int, df: DataFrame):
    pass

def apply_svm(training_data:int, testing_data: int, kernel: str, df: DataFrame):
    pass