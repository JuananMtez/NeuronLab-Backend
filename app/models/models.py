from sqlalchemy import Column, Integer, String, ForeignKey, Float, Enum, Table, Boolean
from sqlalchemy.orm import relationship

from app.config.database import Base
import enum


class NameChannel(enum.Enum):
    NZ = 1
    Fp1 = 2
    FpZ = 3
    Fp2 = 4
    AF7 = 5
    AF3 = 6
    AFZ = 7
    AF4 = 8
    AF8 = 9
    F9 = 10
    F7 = 11
    F5 = 12
    F3 = 13
    Fz = 14
    F2 = 15
    F4 = 16
    F6 = 17
    F8 = 18
    F10 = 19
    FT9 = 20
    FT7 = 21
    FC5 = 22
    FC3 = 23
    FC1 = 24
    FCZ = 25
    FC2 = 26
    FC4 = 27
    FC6 = 28
    FC8 = 29
    FC10 = 30
    A1 = 31
    T9 = 32
    T7 = 33
    C5 = 34
    C3 = 35
    C1 = 36
    Cz = 37
    C2 = 38
    C4 = 39
    C6 = 40
    T8 = 41
    T10 = 42
    A2 = 43
    TP9 = 44
    TP7 = 45
    CP5 = 46
    CP3 = 47
    CP1 = 48
    CPZ = 49
    CP2 = 50
    CP4 = 51
    CP6 = 52
    CP8 = 53
    CP10 = 54
    P9 = 55
    P7 = 56
    P5 = 57
    P3 = 58
    P1 = 59
    Pz = 60
    P2 = 61
    P4 = 62
    P6 = 63
    P8 = 64
    P10 = 65
    PO7 = 66
    PO3 = 67
    POZ = 68
    PO4 = 69
    PO8 = 70
    O1 = 71
    OZ = 72
    O2 = 73
    IZ = 74


class ICAMethod(enum.Enum):
    fastica = 1


class Kernel(enum.Enum):
    linear = 1
    poly = 2
    rbf = 3


class FilterMethod(enum.Enum):
    IIR = 1
    FIR = 2


class Optimizer(enum.Enum):
    SGD = 1


class ActivationFunc(enum.Enum):
    sigmoid = 1
    softmax = 2


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
    password = Column(String(255))

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

    training_models = relationship("TrainingModel")


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


class CSV(Base):
    __tablename__ = 'csv'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    path = Column(String(255), unique=True)
    original = Column(Boolean, default=False)
    subject_name = Column(String(255), index=True)

    experiment_id = Column(Integer, ForeignKey('experiment.id'))

    positions = relationship("Position", cascade="save-update, delete")


class Position(Base):
    __tablename__ = 'position'

    id = Column(Integer, primary_key=True)
    position = Column(Integer)

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

    method = Column("method", Enum(FilterMethod))

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
