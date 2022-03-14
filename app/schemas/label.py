from pydantic import BaseModel


class LabelPost(BaseModel):
    label: str


class LabelResponse(BaseModel):
    label: str

    class Config:
        orm_mode = True