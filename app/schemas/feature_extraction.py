from pydantic import BaseModel


class FeaturePost(BaseModel):
    csvs: list[int]
    feature: str


class FeaturesResponse(BaseModel):
    id: int
    feature_extraction: str


    class Config:
        orm_mode = True

