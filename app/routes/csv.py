from fastapi import APIRouter, Depends, Response, File, UploadFile, Form, HTTPException
from ..config.database import get_db
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND
from app.schemas.csv import CSVResponse, CSVCopy, CSVFilters
from app.schemas.preproccessing import PreproccessingResponse, ICAMethod, ICAExclude
from app.schemas.feature_extraction import FeaturesResponse, FeaturePost
from app.services import csv as csv_service
from fastapi.responses import FileResponse




csv_controller = APIRouter(
    prefix="/csv",
    tags=["csvs"])


@csv_controller.get("/{experiment_id}", response_model=list[CSVResponse])
async def get_all_csv(experiment_id: int, db = Depends(get_db)):
    csvs = csv_service.get_all_csv_experiment(db, experiment_id)
    if csvs is None:
        return Response(status_code=HTTP_404_NOT_FOUND)

    return csvs


@csv_controller.get("/{csv_id}/preproccessing", response_model=list[PreproccessingResponse])
async def get_all_preproccessing(csv_id: int, db = Depends(get_db)):
    preproccessings = csv_service.get_all_csv_preproccessing(db, csv_id)
    if preproccessings is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return preproccessings


@csv_controller.get("/{csv_id}/features", response_model=list[FeaturesResponse])
async def get_all_features(csv_id: int, db = Depends(get_db)):
    features = csv_service.get_all_csv_features(db, csv_id)
    if features is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return features


@csv_controller.post("/", response_model=CSVResponse)
async def create_csv(name: str, subject_id: int, experiment_id: int, time_correction: float, files: list[UploadFile], db = Depends(get_db)):

    c = csv_service.create_csv(db, name, subject_id, experiment_id, time_correction, files)
    if c is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return c


@csv_controller.delete("/{csv_id}")
async def delete_csv(csv_id: int, db=Depends(get_db)):

    if csv_service.delete_csv(db, csv_id) is False:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return Response(status_code=HTTP_204_NO_CONTENT)


@csv_controller.post("/{csv_id}", response_model=CSVResponse)
async def copy_csv(csv_id: int, csv_copy: CSVCopy, db =Depends(get_db)):

    c = csv_service.csv_copy(db, csv_id, csv_copy)
    if c is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return c


@csv_controller.patch("/{csv_id}", response_model=CSVResponse)
async def change_name(csv_id: int, csv_copy: CSVCopy, db =Depends(get_db)):

    c = csv_service.change_name(db, csv_id, csv_copy)
    if c is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return c


@csv_controller.post("/preproccessing/list")
async def apply_preproccessing(csv_filters: CSVFilters, db=Depends(get_db)):
    text = csv_service.apply_preproccessing(db, csv_filters)
    if text is not None:
        raise HTTPException(status_code=500, detail=text)
    return Response(status_code=HTTP_204_NO_CONTENT)


@csv_controller.post("/feature/list")
async def apply_feature(feature_post: FeaturePost, db=Depends(get_db)):
    text = csv_service.apply_feature(db, feature_post)
    if text is not None:
        raise HTTPException(status_code=500, detail=text)
    return Response(status_code=HTTP_204_NO_CONTENT)


@csv_controller.post("/{csv_id}/ica/plot/components")
async def plot_components_ica(csv_id: int, ica_method: ICAMethod, db=Depends(get_db)):
    object = csv_service.plot_components_ica(db, csv_id, ica_method)
    if object is None:
        return Response(status_code=HTTP_404_NOT_FOUND)

    return object

@csv_controller.post("/{csv_id}/ica/plot/properties")
async def plot_properties_ica(csv_id: int, ica_method: ICAMethod, db=Depends(get_db)):
    images = csv_service.plot_properties_ica(db, csv_id, ica_method)
    if images is None:
        return Response(status_code=HTTP_404_NOT_FOUND)

    return images

@csv_controller.post("/{csv_id}/ica/apply")
async def exclude_components(csv_id: int, arg: ICAExclude, db=Depends(get_db)):
    text = csv_service.components_exclude_ica(db, csv_id, arg)
    if text is not None:
        raise HTTPException(status_code=500, detail=text)
    return Response(status_code=HTTP_204_NO_CONTENT)

@csv_controller.get("/{csv_id}/download")
async def download_csv(csv_id: int, db=Depends(get_db)):
    csv = csv_service.get_csv_by_id(db, csv_id)
    if csv is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return FileResponse(csv.path, filename=csv.name+".csv")


@csv_controller.get("/{csv_id}/same", response_model=list[CSVResponse])
async def download_csv(csv_id: int, db=Depends(get_db)):
    csvs = csv_service.get_csvs_same(db, csv_id)
    if csvs is None:
        return Response(status_code=HTTP_404_NOT_FOUND)
    return csvs



