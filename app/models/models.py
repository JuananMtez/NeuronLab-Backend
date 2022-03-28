from sqlalchemy import Column, Integer, String, ForeignKey, Float, Enum, Table, Boolean
from sqlalchemy.orm import relationship

from app.config.database import Base
import enum


class NameChannel(enum.Enum):
    Fp1 = 1
    FpZ = 2
    Fp2 = 3
    AF9 = 4
    AF7 = 5
    AF5 = 6
    AF3 = 7
    AF1 = 8
    AFZ = 9
    AF2 = 10
    AF4 = 11
    AF6 = 12
    AF8 = 13
    AF10 = 14
    F9 = 15
    F7 = 16
    F5 = 17
    F3 = 18
    F1 = 19
    Fz = 20
    F2 = 21
    F4 = 22
    F6 = 23
    F8 = 24
    F10 = 25
    FT9 = 26
    FT7 = 27
    FC5 = 28
    FC3 = 29
    FC1 = 30
    FCZ = 31
    FC2 = 32
    FC4 = 33
    FC6 = 34
    FT8 = 35
    FT10 = 36
    T9 = 37
    T7 = 38
    C5 = 39
    C3 = 40
    C1 = 41
    Cz = 42
    C2 = 43
    C4 = 44
    C6 = 45
    T8 = 46
    T10 = 47
    TP9 = 48
    TP7 = 49
    CP5 = 50
    CP3 = 51
    CP1 = 52
    CPZ = 53
    CP2 = 54
    CP4 = 55
    CP6 = 56
    TP8 = 57
    TP10 = 58
    P9 = 59
    P7 = 60
    P5 = 61
    P3 = 62
    P1 = 63
    Pz = 64
    P2 = 65
    P4 = 66
    P6 = 67
    P8 = 68
    P10 = 69
    PO9 = 70
    PO7 = 71
    PO5 = 72
    PO3 = 73
    PO1 = 74
    POZ = 75
    PO2 = 76
    PO4 = 77
    PO6 = 78
    PO8 = 79
    PO10 = 80
    O1 = 81
    OZ = 82
    O2 = 83
    O9 = 84
    IZ = 85
    O10 = 86
    T3 = 87
    T5 = 88
    T4 = 89
    T6 = 90
    M1 = 91
    M2 = 92
    A1 = 93
    A2 = 94



Researcher_Experiment = Table('researcher_experiment', Base.metadata,
                              Column('researcher_id', ForeignKey('researcher.id'), primary_key=True),
                              Column('experiment_id', ForeignKey('experiment.id'), primary_key=True))

Experiment_Subject = Table('experiment_subject', Base.metadata,
                           Column('experiment_id', ForeignKey('experiment.id'), primary_key=True),
                           Column('subject_id', ForeignKey('subject.id'), primary_key=True))


class Researcher(Base):
    __tablename__ = 'researcher'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    surname = Column(String(255))
    email = Column(String(255), unique=True, index=True)
    user = Column(String(255), unique=True, index=True)
    password = Column(String(255), index=True)

    experiments = relationship(
        "Experiment",
        secondary=Researcher_Experiment,
        back_populates="researchers")


class Experiment(Base):
    __tablename__ = 'experiment'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    description = Column(String(255))
    researcher_creator_id = Column(Integer)

    labels = relationship("Label", cascade="save-update, delete")

    device = relationship("Device", back_populates="experiment", uselist=False, cascade="save-update, delete")

    researchers = relationship(
        "Researcher",
        secondary=Researcher_Experiment,
        back_populates="experiments")

    subjects = relationship(
        "Subject",
        secondary=Experiment_Subject,
        back_populates="experiments")

    csvs = relationship("CSV", cascade="save-update, delete")

    trainings = relationship("Training", cascade="save-update, delete")


class Label(Base):
    __tablename__ = 'label'

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(255))
    description = Column(String(255))
    experiment_id = Column(Integer, ForeignKey('experiment.id'))


class Device(Base):
    __tablename__ = 'device'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    sample_rate = Column(Float)

    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    experiment = relationship("Experiment", back_populates="device")

    type = Column(String(50))

    __mapper_args__ = {
        'polymorphic_identity': 'device',
        'polymorphic_on': type
    }


class EEGHeadset(Device):
    __tablename__ = 'eeg_headset'

    id = Column(Integer, ForeignKey('device.id'), primary_key=True, index=True)
    channels = relationship("Channel", cascade="save-update, delete")
    channels_count = Column(Integer)

    __mapper_args__ = {
        'polymorphic_identity': 'eeg_headset',
    }


class Channel(Base):
    __tablename__ = 'channel'

    id = Column(Integer, primary_key=True, index=True)
    channel = Column("channel", Enum(NameChannel))
    position = Column(Integer)

    eeg_headset_id = Column(Integer, ForeignKey('eeg_headset.id'))


class Subject(Base):
    __tablename__ = 'subject'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    surname = Column(String(255))
    gender = Column(String(10))
    age = Column(Integer)
    total_experiments_performed = Column(Integer)

    mental_conditions = relationship("MentalCondition", cascade="save-update, delete")
    experiments = relationship(
        "Experiment",
        secondary=Experiment_Subject,
        back_populates="subjects")


class MentalCondition(Base):
    __tablename__ = 'mental_condition'

    id = Column(Integer, primary_key=True, index=True)
    condition = Column(String(255), index=True)
    subject_id = Column(Integer, ForeignKey('subject.id'))


CSV_Training = Table('csv_training', Base.metadata,
         Column('csv_id', ForeignKey('csv.id'), primary_key=True),
         Column('training_id', ForeignKey('training.id'), primary_key=True))


class CSV(Base):
    __tablename__ = 'csv'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    path = Column(String(255), unique=True)
    type = Column(String(255), index=True)
    subject_name = Column(String(255), index=True)

    experiment_id = Column(Integer, ForeignKey('experiment.id'))

    preproccessing_list = relationship("Preproccessing", cascade="save-update, delete")
    feature_extractions = relationship("FeatureExtraction", cascade="save-update, delete")

    trainings = relationship(
        "Training",
        secondary=CSV_Training,
        back_populates="csvs")


class Preproccessing(Base):
    __tablename__ = 'preproccessing'

    id = Column(Integer, primary_key=True, index=True)
    position = Column(Integer, index=True)
    preproccessing = Column(String(255))
    description = Column(String(255))
    csv_id = Column(Integer, ForeignKey('csv.id'))


class FeatureExtraction(Base):
    __tablename__ = 'feature_extraction'

    id = Column(Integer, primary_key=True, index=True)
    feature_extraction = Column(String(255))
    position = Column(Integer, unique=True)
    description = Column(String(255))

    csv_id = Column(Integer, ForeignKey('csv.id'))


class Training(Base):
    __tablename__ = 'training'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    description = Column(String(500))
    preproccesing_description = Column(String(255), index=True)
    path = Column(String(255), unique=True)
    experiment_id = Column(Integer, ForeignKey('experiment.id'))

    csvs = relationship(
        "CSV",
        secondary=CSV_Training,
        back_populates="trainings")





'''

class Position(Base):
    __tablename__ = 'position'

    id = Column(Integer, primary_key=True)
    position = Column(Integer)
    specification = Column(String(255))
    csv_id = Column(Integer, ForeignKey('csv.id'))
    filter = relationship("Filter", back_populates="position", uselist=False)
    downsampling = relationship("Downsampling", back_populates="position", uselist=False)
    ica = relationship("ICA", back_populates="position", uselist=False)
    feature_extraction = relationship("FeatureExtraction", back_populates="position", uselist=False)


class Downsampling(Base):
    __tablename__ = 'downsampling'

    id = Column(Integer, primary_key=True, index=True)
    freq = Column(Float)

    position_id = Column(Integer, ForeignKey('position.id'))
    position = relationship("Position", back_populates="downsampling")


class ICA(Base):
    __tablename__ = 'ica'

    id = Column(Integer, primary_key=True, index=True)
    method = Column("method", Enum(ICAMethod))

    labels = relationship("ComponentRemoved")

    position_id = Column(Integer, ForeignKey('position.id'))
    position = relationship("Position", back_populates="ica")




class ComponentRemoved(Base):
    __tablename__ = 'component_removed'

    id = Column(Integer, primary_key=True, index=True)
    component = Column(Integer)
    ica_id = Column(Integer, ForeignKey('ica.id'))


class FeatureExtraction(Base):
    __tablename__ = 'feature_extraction'

    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(Integer, ForeignKey('position.id'))
    position = relationship("Position", back_populates="feature_extraction")


class Filter(Base):
    __tablename__ = 'filter'

    id = Column(Integer, primary_key=True, index=True)
    phase = Column(Integer)
    order = Column(Integer)

    position_id = Column(Integer, ForeignKey('position.id'))
    position = relationship("Position", back_populates="filter")

    method = Column(String(255))

    type = Column(String(50))

    __mapper_args__ = {
        'polymorphic_identity': 'filter',
        'polymorphic_on': type
    }


class LowPass(Filter):
    __tablename__ = 'low_pass_filter'

    id = Column(Integer, ForeignKey('filter.id'), primary_key=True)
    low_freq = Column(Float)
    __mapper_args__ = {
        'polymorphic_identity': 'low_pass_filter',
    }


class HighPass(Filter):
    __tablename__ = 'high_pass_filter'

    id = Column(Integer, ForeignKey('filter.id'), primary_key=True)
    high_freq = Column(Float)
    __mapper_args__ = {
        'polymorphic_identity': 'high_pass_filter',
    }


class BandPass(Filter):
    __tablename__ = 'band_pass_filter'

    id = Column(Integer, ForeignKey('filter.id'), primary_key=True)
    low_freq = Column(Float)
    high_freq = Column(Float)
    __mapper_args__ = {
        'polymorphic_identity': 'band_pass_filter',
    }


class Notch(Filter):
    __tablename__ = 'notch_filter'

    id = Column(Integer, ForeignKey('filter.id'), primary_key=True)
    freqs = relationship("FrequenciesNotch")
    __mapper_args__ = {
        'polymorphic_identity': 'notch_filter',
    }


class FrequenciesNotch(Base):
    __tablename__ = 'frequencies_notch'

    id = Column(Integer, primary_key=True, index=True)
    freq = Column(Float)
    notch_id = Column(Integer, ForeignKey('notch_filter.id'))


class TrainingModel(Base):
    __tablename__ = 'training_model'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    path = Column(String(255), unique=True)
    training_data = Column(Float)
    test_data = Column(Float)

    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    type = Column(String(50))

    __mapper_args__ = {
        'polymorphic_identity': 'training_model',
        'polymorphic_on': type
    }


class KNN(TrainingModel):
    __tablename__ = 'knn'

    id = Column(Integer, ForeignKey('training_model.id'), primary_key=True)
    num_neighbours = Column(Integer)
    __mapper_args__ = {
        'polymorphic_identity': 'knn',
    }


class RandomForest(TrainingModel):
    __tablename__ = 'random_forest'

    id = Column(Integer, ForeignKey('training_model.id'), primary_key=True)
    max_depth = Column(Integer)
    n_estimators = Column(Integer)
    random_state = Column(Integer)
    __mapper_args__ = {
        'polymorphic_identity': 'random_forest',
    }


class SVM(TrainingModel):
    __tablename__ = 'svm'

    id = Column(Integer, ForeignKey('training_model.id'), primary_key=True)
    kernel = Column("kernel", Enum(Kernel))
    __mapper_args__ = {
        'polymorphic_identity': 'svm',
    }


class DeepLearning(TrainingModel):
    __tablename__ = "deep_learning"

    id = Column(Integer, ForeignKey('training_model.id'), primary_key=True)
    optimizer = Column("optimizer", Enum(Optimizer))
    learning_rate = Column(Float)
    layers = relationship("Layer")

    __mapper_args__ = {
        'polymorphic_identity': 'deep_learning',
    }


class Layer(Base):
    __tablename__ = "layer"

    id = Column(Integer, primary_key=True)
    num_neurons = Column(Integer)
    activation_function = Column("optimizer", Enum(ActivationFunc))
    deep_learning_id = Column(Integer, ForeignKey('deep_learning.id'))
    type = Column(String(50))

    __mapper_args__ = {
        'polymorphic_identity': 'layer',
        'polymorphic_on': type
    }


class Dense(Layer):
    __tablename__ = 'dense'

    id = Column(Integer, ForeignKey('layer.id'), primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'dense',
    }


class Convolutional(Layer):
    __tablename__ = 'convolutional'

    id = Column(Integer, ForeignKey('layer.id'), primary_key=True)
    inputs_shape = relationship("InputShape")
    __mapper_args__ = {
        'polymorphic_identity': 'convolutional',
    }


class InputShape(Base):
    __tablename__ = 'input_shape'

    id = Column(Integer, primary_key=True, index=True)
    data = Column(Integer)
    convolutional_id = Column(Integer, ForeignKey('convolutional.id'))
'''