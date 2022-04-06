from typing import Optional
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from sqlalchemy.orm import Session
import io
import os
import app.crud.csv as csv_crud
import app.crud.training as training_crud
from app.models import models
from app.schemas.training import MachineLearningPost, DeepLearningPost
from datetime import datetime
from joblib import dump, load
import app.crud.experiment as experiment_crud
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import Dense
import tensorflow as tf


def create_training_machine(db: Session, training_post: MachineLearningPost):
    exp = experiment_crud.find_by_id(db, training_post.exp_id)
    if exp is None:
        return None

    clf = None
    description = ''
    db_training = models.Training(
        name=training_post.name,
        experiment_id=training_post.exp_id,
        type='Machine Learning')

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
    labels = np.array([df['Stimulus'].tolist()]).T

    del df['Stimulus']

    rawdata = df.values

    del df


    if training_post.algorithm.__class__.__name__ == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=training_post.algorithm.n_neighbors)
        description = "KNN (n_neighbors: " + str(training_post.algorithm.n_neighbors) + ")"

    elif training_post.algorithm.__class__.__name__ == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=training_post.algorithm.n_estimators, max_depth=training_post.algorithm.max_depth, random_state=training_post.algorithm.random_state)
        description = "Random Forest (max_depth: " + str(training_post.algorithm.max_depth) + ", n_estimatos: " + str(training_post.algorithm.n_estimators) + ", random_state: " + str(training_post.algorithm.random_state) + ")"

    elif training_post.algorithm.__class__.__name__ == 'SVM':
        clf = SVC(kernel=training_post.algorithm.kernel)
        description = "SVM (kernel: " + training_post.algorithm.kernel + ")"

    description += "\nTraining Data = " + str(training_post.training_data) + "%, Testing Data = " + str(training_post.testing_data) + "%"
    X_train, X_test, y_train, y_test = train_test_split(rawdata, labels, test_size=training_post.testing_data/100, random_state=0, shuffle=True)

    clf.fit(X=X_train, y=y_train)

    name_model = generate_name_model('machine')

    db_training.validation = str(classification_report(y_test, clf.predict(X_test)))
    db_training.path = name_model
    db_training.description = description

    dump(clf, db_training.path)

    training_crud.save(db, db_training)


def delete_training(db: Session, training_id: int):
    training = training_crud.find_by_id(db, training_id)
    os.remove(training.path)

    training_crud.delete(db, training)


def find_all_csv(db: Session, csv_id: int) -> Optional[list[models.Training]]:
    csv = csv_crud.find_by_id(db, csv_id)
    if csv is None:
        return None
    return csv.trainings


def generate_name_model(type: str):
    now = datetime.now()
    if type == 'machine':
        return "models/record_{}.joblib".format(now.strftime("%d-%m-%Y-%H-%M-%S"))
    elif type == 'deep':
        return "models/record_{}.h5".format(now.strftime("%d-%m-%Y-%H-%M-%S"))



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
    labels = np.array([df['Stimulus'].tolist()]).T
    del df['Stimulus']

    rawdata = df.values
    del df

    if training.type == 'Machine Learning':

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

    elif training.type == 'Deep Learning':
        model = keras.models.load_model(training.path)
        try:
            loss, accuracy = model.evaluate(x=rawdata, y=labels, verbose=0)
            text = "Loss: " + str(loss) +"\nAccuracy " + str(accuracy)
            return {"text": text, "n_jumps": 2}
        except tf.errors.InvalidArgumentError as e:
            return str(e)


def create_training_deep(db: Session, training_post: DeepLearningPost):
    exp = experiment_crud.find_by_id(db, training_post.exp_id)
    if exp is None:
        return None

    description = 'Optimizer: ' + training_post.optimizer.capitalize() + ' Loss: ' + training_post.loss.capitalize() + ", "
    if training_post.type == 'manual':
        description += "Learning Rate: " + str(training_post.learning_rate) + "\n"
    else:
        description += "Learning Rate: default" + "\n"

    db_training = models.Training(
        name=training_post.name,
        experiment_id=training_post.exp_id,
        type='Deep Learning'
    )

    model = Sequential()
    if len(training_post.layers) > 0:
        description += " Layer 1 - " + \
                        training_post.layers[0].type.capitalize() + ", " + \
                       "Number of neurons: " + str(training_post.layers[0].num_neurons).capitalize() + ", " + \
                        " Activation Function: " + training_post.layers[0].activation_func.capitalize()+ ", " + \
                        " Kernel Initializer: " + training_post.layers[0].kernel_initializer.capitalize() + ", " + \
                        "Input Shape: (" + training_post.layers[0].batch_size + \
                       ", " + training_post.layers[0].input_size + ")\n"

        if training_post.layers[0].input_size == '':

            model.add(Dense(training_post.layers[0].num_neurons,
                  kernel_initializer=training_post.layers[0].kernel_initializer,
                  activation=training_post.layers[0].activation_func,
                  input_shape=(int(training_post.layers[0].batch_size),)))
        else:
            model.add(Dense(training_post.layers[0].num_neurons,
                  kernel_initializer=training_post.layers[0].kernel_initializer,
                  activation=training_post.layers[0].activation_func,
                  input_shape=(int(training_post.layers[0].batch_size),int(training_post.layers[0].input_size))))

        training_post.layers = training_post.layers[1:]

    for i in range(len(training_post.layers)):
        description += " Layer " + str(i+2) + " - " + \
                        training_post.layers[0].type.capitalize() + ", " + \
                       "Number of neurons: " + str(training_post.layers[i].num_neurons) + ", " + \
                        " Activation Function: " + training_post.layers[i].activation_func.capitalize() + "\n"
        model.add(Dense(training_post.layers[i].num_neurons, activation=training_post.layers[i].activation_func))

    description += "\nTraining Data = " + str(training_post.training_data) + "%, Testing Data = " + str(training_post.testing_data) + "%"

    opt = None
    if training_post.type == 'default':
        if training_post.optimizer == 'sgd':
            opt = optimizers.gradient_descent_v2.SGD()
        elif training_post.optimizer == 'adam':
            opt = optimizers.adam_v2.Adam()
    else:
        if training_post.optimizer == 'sgd':
            opt = optimizers.gradient_descent_v2.SGD(learning_rate=training_post.learning_rate)
        elif training_post.optimizer == 'adam':
            opt = optimizers.adam_v2.Adam(learning_rate=training_post.learning_rate)


    model.compile(loss=training_post.loss, optimizer=opt, metrics=['accuracy'])


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
    df = df.sample(n=int((training_post.training_data * df.shape[0]) / 100))

    labels = np.array([df['Stimulus'].tolist()]).T
    del df['Stimulus']
    rawdata = df.values
    del df

    oldStdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()


    X_train, X_test, y_train, y_test = train_test_split(rawdata, labels, test_size=training_post.testing_data/100, random_state=0, shuffle=True)

    try:
        model.fit(x=X_train, y=y_train, epochs=training_post.epochs, verbose=1)
        sys.stdout = oldStdout

        loss, accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)
        text = "Loss: " + str(loss) + ", Accuracy " + str(accuracy)
        db_training.validation = mystdout.getvalue() + "\n\n" + text

    except ValueError as e:
        return str(e)

    name_model = generate_name_model('deep')
    db_training.path = name_model
    db_training.description = description

    model.save(name_model)
    training_crud.save(db, db_training)

    return True


def get_summary(db: Session, training_id: int):

    training = training_crud.find_by_id(db, training_id)
    if training is None:
        return None

    model = keras.models.load_model(training.path)
    text = get_model_summary(model)

    cont = 0
    for x in text:
        if x == '\n':
            cont += 1

    return {"text": text, "n_jumps": cont}

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def get_all_csvs(db: Session, training_id: int) -> list[models.CSV]:
    training = training_crud.find_by_id(db, training_id)
    return training.csvs