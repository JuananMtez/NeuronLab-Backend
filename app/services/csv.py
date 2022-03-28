from typing import Optional, Any

from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.models import models
from app.crud import csv as csv_crud
from app.crud import experiment as experiment_crud
from app.crud import subject as subject_crud
from datetime import datetime, timedelta
import os
from app.schemas.csv import CSVCopy, CSVFilters
from app.schemas.preproccessing import ICAMethod, ICAExclude
from app.schemas.feature_extraction import FeaturePost
import json
import numpy as np
import pandas as pd
from mne.io import RawArray
import mne
import base64
import app.crud.training as training_crud


def get_csv_by_id(db: Session, csv_id: int) -> Optional[models.CSV]:
    csv = csv_crud.find_by_id(db, csv_id)
    return csv


def get_all_csv_preproccessing(db: Session, csv_id: int) -> Optional[list[models.Preproccessing]]:
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    return csv.preproccessing_list


def get_all_csv_features(db: Session, csv_id: int) -> Optional[list[models.FeatureExtraction]]:
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    return csv.feature_extractions

def get_all_csv_experiment(db: Session, experiment_id: int) -> Optional[list[models.Experiment]]:
    e = experiment_crud.find_by_id(db, experiment_id)
    if e is None:
        return None
    return e.csvs


def create_csv(db: Session, name: str, subject_id: int, experiment_id: int, time_correction: float, files: list[UploadFile]) -> Optional[models.CSV]:
    exp = experiment_crud.find_by_id(db, experiment_id)
    subject = subject_crud.find_by_id(db, subject_id)
    if exp is None or subject is None:
        return None

    object = {
        "dataInput": [],
        "timestamp": [],
        "stimuli": []
    }
    name_file = generate_name_csv(db)

    for file in files:
        aux = json.loads(file.file.read())
        object["dataInput"].extend(aux["dataInput"])
        object["timestamp"].extend(aux["timestamp"])
        object["stimuli"].extend(aux["stimuli"])

    for stimulus in object["stimuli"]:
        cont = 0
        for label in exp.labels:
            if stimulus[0][0] != int(label.label):
                cont += 1
        if cont == len(exp.labels):
            return None

    if exp.device.type == 'eeg_headset':
        create_csv_eegheadset(object, exp, name_file, time_correction)

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
    for training in csv.trainings:
        if len(training.csvs) == 1:
            training_crud.delete(db, training)

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

    name_file = generate_name_csv(db)
    file.to_csv(name_file, index=False)

    db_csv = models.CSV(name=csv_copy.name,
                        subject_name=csv_original.subject_name,
                        type='copied',
                        experiment_id=csv_original.experiment_id,
                        path=name_file)

    for x in csv_original.preproccessing_list:
        db_preproccessing = models.Preproccessing(
            position=x.position,
            preproccessing=x.preproccessing,
            csv_id=db_csv.id,
            description=x.description)
        db_csv.preproccessing_list.append(db_preproccessing)

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
            if len(csv.feature_extractions) > 0:
                return "In some csv have already applied feature extraction. Please, unselect"
            if exp is None:
                exp = experiment_crud.find_by_id(db, csv.experiment_id)

            df = pd.read_csv(csv.path)
            rawdata = load_raw(df, exp)
            name_file = generate_name_csv(db)
            if csv.type == 'copied' or csv.type == 'copied | preproccessed':
                type = 'copied | preproccessed'
            else:
                type = 'preproccessed'



            new_csv = models.CSV(
                            name=csv.name + " (p)",
                            subject_name=csv.subject_name,
                            path=name_file,
                            type=type,
                            experiment_id=csv.experiment_id)

            for x in csv.preproccessing_list:
                db_preproccessing = models.Preproccessing(
                    position=x.position,
                    preproccessing=x.preproccessing,
                    csv_id=new_csv.id,
                    description=x.description)
                new_csv.preproccessing_list.append(db_preproccessing)

            if rawdata is not None:
                for prep in csv_filters.preproccessings:
                    if prep.__class__.__name__ == 'CSVBandpass':
                        try:
                            rawdata = apply_bandpass(prep, rawdata, new_csv)

                        except ValueError:
                            return "Check frequency values"
                        except np.linalg.LinAlgError:
                            return "Array must not contain infs or NaNs"
                    elif prep.__class__.__name__ == 'CSVNotch':
                        try:
                            rawdata = apply_notch(prep, rawdata, new_csv)
                        except ValueError:
                            return "Check frequency values"
                        except np.linalg.LinAlgError:
                            return "Array must not contain infs or NaNs"
                    elif prep.__class__.__name__ == 'CSVDownsampling':
                        try:
                            rawdata = apply_downsampling(prep, rawdata, new_csv)
                        except ValueError:
                            return "Check frequency values"
                        except np.linalg.LinAlgError:
                            return "Array must not contain infs or NaNs"

                ch_names = []
                for x in exp.device.channels:
                    ch_names.append(x.channel.name)

                data = convert_to_df(rawdata, ch_names)
                data.to_csv(new_csv.path, index=False)

                csv_crud.save(db, new_csv)





def load_raw(data, experiment):

    if experiment.device.type == 'eeg_headset':
        if "Timestamp" in data.columns:
            del data['Timestamp']
        ch_names = list(data.columns)[0:experiment.device.channels_count] + ['Stim']
        ch_types = ['eeg'] * experiment.device.channels_count + ['stim']

        ch_ind = []
        for x in range(0, experiment.device.channels_count):
            ch_ind.append(x)

        data = data.values[:, ch_ind + [experiment.device.channels_count]].T
        #data[:-1] *= 1e-6



        info = mne.create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=experiment.device.sample_rate)
        raw = RawArray(data=data, info=info)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)

        return raw

    return None


def apply_feature(db: Session, feature_post: FeaturePost):
    exp = None
    data = None
    for csv_id in feature_post.csvs:
        csv = csv_crud.find_by_id(db, csv_id)
        if csv is None:
            break
        if exp is None:
            exp = experiment_crud.find_by_id(db, csv.experiment_id)

        df = pd.read_csv(csv.path)
        name_file = generate_name_csv(db)

        if csv.type == 'copied':
            type = 'copied | feature'
        elif csv.type == 'copied | preproccessed':
            type = 'copied | preproccessed | feature'
        elif csv.type == 'preproccessed':
            type = 'preproccessed | feature'
        else:
            type = 'feature'

        new_csv = models.CSV(
            name=csv.name + " (f)",
            subject_name=csv.subject_name,
            path=name_file,
            type=type,
            experiment_id=csv.experiment_id)

        for x in csv.preproccessing_list:
            db_preproccessing = models.Preproccessing(
                position=x.position,
                preproccessing=x.preproccessing,
                csv_id=new_csv.id,
                description=x.description)
            new_csv.preproccessing_list.append(db_preproccessing)

        for x in csv.feature_extractions:
            db_f = models.FeatureExtraction(
                position=x.position,
                csv_id=new_csv.id,
                description=x.description,
                feature_extraction=x.feature_extraction)
            new_csv.feature_extractions.append(db_f)


        if feature_post.feature == 'mean':
            data = apply_mean(exp, df)
            db_f = models.FeatureExtraction(
                position=len(new_csv.feature_extractions)+1,
                csv_id=new_csv.id,
                description="Mean",
                feature_extraction="Mean")
            new_csv.feature_extractions.append(db_f)

        elif feature_post.feature == 'variance':
            data = apply_variance(exp, df)
            db_f = models.FeatureExtraction(
                position=len(new_csv.feature_extractions) + 1,
                csv_id=new_csv.id,
                description="Variance",
                feature_extraction="Variance")


        data.to_csv(name_file, index=False)



        csv_crud.save(db, new_csv)



def generate_name_csv(db: Session):
    now = datetime.now()
    name_file = "csvs/record_{}.csv".format(now.strftime("%d-%m-%Y-%H-%M-%S"))
    while csv_crud.find_by_path(db, name_file):
        now = datetime.now() + timedelta(seconds=1)
        name_file = "csvs/record_{}.csv".format(now.strftime("%d-%m-%Y-%H-%M-%S"))

    return name_file

def generate_name_tmp():
    now = datetime.now()
    return "tmp/record_{}.png".format(now.strftime("%d-%m-%Y-%H-%M-%S"))


def create_csv_eegheadset(obj: Any, exp: models.Experiment, name_file: str, time_correction: float):
    ch_names = []
    for x in exp.device.channels:
        ch_names.append(x.channel.name)

    obj["dataInput"] = np.concatenate(obj["dataInput"], axis=0)
    if len(ch_names) > 8:
        obj["dataInput"] = obj["dataInput"][:, :8]

    obj["timestamp"] = np.array(obj["timestamp"]) + time_correction
    obj["dataInput"] = np.c_[obj["timestamp"], obj["dataInput"]]
    data = pd.DataFrame(data=obj["dataInput"], columns=['Timestamp'] + ch_names)

    if len(obj["stimuli"]) != 0:
        #data['Stimulus'] = 0
        data.loc[:, 'Stimulus'] = 0
        for estim in obj["stimuli"]:
            abs = np.abs(estim[1] - obj["timestamp"])
            ix = np.argmin(abs)
            data.loc[ix, 'Stimulus'] = estim[0][0]

    data.to_csv(name_file, index=False)


def apply_bandpass(prep, rawdata, new_csv):
    l_freq = None
    h_freq = None
    text = ''

    if prep.low_freq != '':
        l_freq = float(prep.low_freq)
        text = text + 'Low Frequency: ' + prep.low_freq + 'Hz '

    if prep.high_freq != '':
        h_freq = float(prep.high_freq)
        text = text + 'High Frequency: ' + prep.high_freq + 'Hz '


    if prep.filter_method == 'fir':
        db_preproccessing = models.Preproccessing(
                                position=len(new_csv.preproccessing_list) + 1,
                                preproccessing='Bandpass',
                                csv_id=new_csv.id,
                                description='Method: FIR, ' + 'Phase: ' + prep.phase + ', ' + text)
        new_csv.preproccessing_list.append(db_preproccessing)

        return rawdata.copy().filter(l_freq=l_freq, h_freq=h_freq,
                                      method='fir', fir_design='firwin', phase=prep.phase)

    elif prep.filter_method == 'iir':
        if prep.order == '1':
            ordinal = 'st'
        elif prep.order == '2':
            ordinal = 'nd'
        else:
             ordinal = 'th'

        db_preproccessing = models.Preproccessing(
                                position=len(new_csv.preproccessing_list) + 1,
                                preproccessing='Bandpass',
                                csv_id=new_csv.id,
                                description='Method: IIR, ' + prep.order + ordinal + '-order Butterworth filter, ' + text)
        new_csv.preproccessing_list.append(db_preproccessing)

        iir_params = dict(order=int(prep.order), ftype='butter')
        return rawdata.copy().filter(l_freq=l_freq, h_freq=h_freq,
                                      method='iir', iir_params=iir_params)


def apply_notch(prep, rawdata, new_csv):

    if prep.filter_method == 'fir':
        db_preproccessing = models.Preproccessing(
                                position=len(new_csv.preproccessing_list) + 1,
                                preproccessing='Notch',
                                csv_id=new_csv.id,
                                description='Method: FIR, ' + 'Phase: ' + prep.phase + ', ' + 'Frequency: ' + prep.freq + 'Hz')
        new_csv.preproccessing_list.append(db_preproccessing)

        return rawdata.copy().notch_filter(freqs=float(prep.freq), method='fir', fir_design='firwin', phase=prep.phase)

    elif prep.filter_method == 'iir':
        if prep.order == '1':
            ordinal = 'st'
        elif prep.order == '2':
            ordinal = 'nd'
        else:
             ordinal = 'th'

        db_preproccessing = models.Preproccessing(
            position=len(new_csv.preproccessing_list) + 1,
            preproccessing='Bandpass',
            csv_id=new_csv.id,
            description='Method: IIR, ' + prep.order + ordinal + '-order Butterworth filter, ' + 'Frequency: ' + prep.freq + 'Hz')
        new_csv.preproccessing_list.append(db_preproccessing)

        iir_params = dict(order=int(prep.order), ftype='butter')
        return rawdata.copy().notch_filter(freqs=float(prep.freq), method='iir', iir_params=iir_params)


def apply_downsampling(prep, rawdata, new_csv):
    db_preproccessing = models.Preproccessing(
        position=len(new_csv.preproccessing_list) + 1,
        preproccessing='Downsampling',
        csv_id=new_csv.id,
        description='Sample rate: ' + prep.freq_downsampling + ' Hz')
    new_csv.preproccessing_list.append(db_preproccessing)

    return rawdata.copy().resample(prep.freq_downsampling, npad="auto")


def convert_to_df(rawdata, ch_names) -> pd.DataFrame:
    data = pd.DataFrame(data=rawdata.get_data().T, columns=ch_names + ['Stimulus'])
    scalar = pd.to_numeric(data['Stimulus'], errors='coerce', downcast='integer')
    del data['Stimulus']
    data['Stimulus'] = scalar
    return data



def plot_properties_ica(db: Session, csv_id, ica_method: ICAMethod):
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    exp = experiment_crud.find_by_id(db, csv.experiment_id)
    if exp is None:
        return None

    df = pd.read_csv(csv.path)

    rawdata = load_raw(df, exp)
    fit_params = None
    if ica_method.method == 'picard':
        fit_params = dict(ortho=True, extended=True)
    elif ica_method.method == 'infomax':
        fit_params = dict(extended=True)

    ica = mne.preprocessing.ICA(random_state=97, method=ica_method.method, fit_params=fit_params)
    ica.fit(rawdata)
    shape = ica.get_components()
    picks = []
    for x in range(0, shape.shape[1]):
        picks.append(x)

    figures = ica.plot_properties(rawdata.copy(), picks=picks)
    returned = []
    for x in figures:
        name_tmp = generate_name_tmp()
        x.savefig(name_tmp)
        with open(name_tmp, 'rb') as f:
            returned.append(base64.b64encode(f.read()))
            os.remove(name_tmp)

    return returned


def plot_components_ica(db: Session, csv_id: int, ica_method: ICAMethod):
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    exp = experiment_crud.find_by_id(db, csv.experiment_id)
    if exp is None:
        return None

    df = pd.read_csv(csv.path)

    rawdata = load_raw(df, exp)
    fit_params = None
    if ica_method.method == 'picard':
        fit_params = dict(ortho=True, extended=True)
    elif ica_method.method == 'infomax':
        fit_params = dict(extended=True)

    ica = mne.preprocessing.ICA(random_state=97, method=ica_method.method, fit_params=fit_params)
    ica.fit(rawdata)
    figure = ica.plot_components()

    name_tmp = generate_name_tmp()
    figure[0].savefig(name_tmp)
    shape = ica.get_components()
    with open(name_tmp, 'rb') as f:
        base64image = base64.b64encode(f.read())
    os.remove(name_tmp)
    returned = {"img": base64image, "components": shape.shape[1]}
    return returned


def components_exclude_ica(db: Session, csv_id: int, arg: ICAExclude):
    csv = csv_crud.find_by_id(db, csv_id)

    if csv is None:
        return None

    exp = experiment_crud.find_by_id(db, csv.experiment_id)
    if exp is None:
        return None

    df = pd.read_csv(csv.path)

    rawdata = load_raw(df, exp)
    fit_params = None
    if arg.method == 'picard':
        fit_params = dict(ortho=True, extended=True)
    elif arg.method == 'infomax':
        fit_params = dict(extended=True)

    ica = mne.preprocessing.ICA(random_state=97, method=arg.method, fit_params=fit_params)
    ica.fit(rawdata)

    ica.exclude = arg.components
    ica.apply(rawdata.copy())

    if csv.type == 'copied' or csv.type == 'copied | preproccessed':
        type = 'copied | preproccessed'
    else:
        type = 'preproccessed'

    name_file = generate_name_csv(db)


    new_csv = models.CSV(
        name=csv.name + " (p)",
        subject_name=csv.subject_name,
        path=name_file,
        type=type,
        experiment_id=csv.experiment_id)

    for x in csv.preproccessing_list:
        db_preproccessing = models.Preproccessing(
            position=x.position,
            preproccessing=x.preproccessing,
            csv_id=new_csv.id,
            description=x.description)
        new_csv.preproccessing_list.append(db_preproccessing)

    text = 'Components removed: '
    for x in arg.components:
        text = text + str(x) + ", "

    text = text[:-1]
    text = text[:-1]

    db_preproccessing = models.Preproccessing(
        position=len(new_csv.preproccessing_list) + 1,
        preproccessing='ICA',
        csv_id=new_csv.id,
        description=text)
    new_csv.preproccessing_list.append(db_preproccessing)

    ch_names = []
    for x in exp.device.channels:
        ch_names.append(x.channel.name)
    data = convert_to_df(rawdata, ch_names)
    data.to_csv(new_csv.path, index=False)

    csv_crud.save(db, new_csv)


def get_csvs_same(db: Session, csv_id:int)-> Optional[list[models.CSV]]:
    csv = csv_crud.find_by_id(db, csv_id)
    if csv is None:
        return None

    all_csv = csv_crud.find_all(db)
    returned = []

    for c in all_csv:
        same = True
        if len(c.preproccessing_list) == len(csv.preproccessing_list) and c.id != csv.id:
            i = 0
            while i < len(c.preproccessing_list) and same == True:
                if c.preproccessing_list[i].description == csv.preproccessing_list[i].description:
                    i = i + 1
                else:
                    same = False
        else:
            same = False

        if same == True:
            if len(c.feature_extractions) == len(csv.feature_extractions):
                i = 0
                while i < len(c.feature_extractions) and same == True:
                    if c.feature_extractions[i].description == csv.feature_extractions[i].description:
                        i = i + 1
                    else:
                        same = False
            else:
                same = False

        if same == True:
            returned.append(c)

    return returned


def apply_mean(exp, df):

    if exp.device.type == 'eeg_headset':
        if "Timestamp" in df.columns:
            del df['Timestamp']

        stimulus = df['Stimulus'].tolist()
        del df['Stimulus']
        df['Mean'] = df.iloc[:, 0:exp.device.channels_count].mean(axis=1)
        df['Stimulus'] = stimulus
    return df

def apply_variance(exp, df):
    if exp.device.type == 'eeg_headset':
        if "Timestamp" in df.columns:
            del df['Timestamp']

        stimulus = df['Stimulus'].tolist()
        del df['Stimulus']
        df['Variance'] = df.iloc[:, 0:exp.device.channels_count].var(axis=1)
        df['Stimulus'] = stimulus
    return df