from fastapi import APIRouter, Depends, Response, File, UploadFile, Form
from ..config.database import get_db
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND
from app.schemas.csv import CSVResponse, CSVCopy, CSVFilters
from app.schemas.preproccessing import PreproccessingResponse
from app.services import csv as csv_service



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


@csv_controller.post("/", response_model=CSVResponse)
async def create_csv(name: str, subject_id: int, experiment_id: int, files: list[UploadFile], db = Depends(get_db)):

    c = csv_service.create_csv(db, name, subject_id, experiment_id, files)
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
    csv_service.apply_preproccessing(db, csv_filters)
    return {'msg', 'ok'}