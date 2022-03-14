from pydantic import BaseModel
from app.schemas.channel import ChannelPost, ChannelResponse


class Device(BaseModel):
    name: str
    sample_rate: int
    type: str


class EEGHeadset(Device):
    channels: list[ChannelResponse]


class EEGHeadsetPost(Device):
    channels: list[ChannelPost]


class EEGHeadsetResponse(EEGHeadset):
    id: int

    class Config:
        orm_mode = True


class DeviceResponse(Device):
    id: int

    class Config:
        orm_mode = True