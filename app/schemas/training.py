from pydantic import BaseModel
from typing import Union




class KNN(BaseModel):
    n_neighbors: int


class RandomForest(BaseModel):
    max_depth: int
    n_estimators: int
    random_state: int


class SVM(BaseModel):
    kernel: str


algorithm_machine = Union[KNN, RandomForest, SVM]


class DeepLearning(BaseModel):
    pass


class MachineLearningPost(BaseModel):
    csvs: list[int]
    name: str
    testing_data: int
    training_data: int
    algorithm: algorithm_machine


class TrainingResponse(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        orm_mode = True