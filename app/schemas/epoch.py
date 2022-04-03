from pydantic import BaseModel


class EpochPlot(BaseModel):
    n_events: int


class EpochAverage(BaseModel):
    channel: str
    label: str


class EpochCompare(BaseModel):
    label: str


class EpochActivity(BaseModel):
    label: str
    times: list[float]
    extrapolate: str