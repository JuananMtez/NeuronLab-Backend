from pydantic import BaseModel


class FeaturePost(BaseModel):
    csvs: list[int]
    feature: str


class FeaturesResponse(BaseModel):
    id: int
    position: int
    feature_extraction: str
    description: str

    class Config:
        orm_mode = True

