from pydantic import BaseModel


class LabelPost(BaseModel):
    label: str
    description: str


class LabelResponse(BaseModel):
    label: str
    description: str

    class Config:
        orm_mode = True