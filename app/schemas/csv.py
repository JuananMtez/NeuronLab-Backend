from pydantic import BaseModel


class CSVCopy(BaseModel):
    name: str


class CSVResponse(BaseModel):
    id: int
    name: str
    original: bool
    subject_name: str

    class Config:
        orm_mode = True