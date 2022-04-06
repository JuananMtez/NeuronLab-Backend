from pydantic import BaseModel


class EpochPlot(BaseModel):
    n_events: int


class EpochAverage(BaseModel):
    channel: str
    label: str


class EpochCompare(BaseModel):
    label: str


class EpochPSD(BaseModel):
    f_min: float
    f_max: float
    average: bool

class EpochActivity(BaseModel):
    label: str
    times: list[float]
    extrapolate: str