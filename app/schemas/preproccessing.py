from pydantic import BaseModel


class PreproccessingResponse(BaseModel):
    id: int
    position: int
    preproccessing: str
    description: str

    class Config:
        orm_mode = True