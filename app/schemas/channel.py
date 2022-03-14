from pydantic import BaseModel
from app.models.models import NameChannel


class ChannelPost(BaseModel):
    channel: NameChannel


class ChannelResponse(BaseModel):
    channel: NameChannel

    class Config:
        orm_mode = True