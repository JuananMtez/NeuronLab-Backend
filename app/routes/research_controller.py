from fastapi import APIRouter


research_controller = APIRouter(
    prefix="/research",
    tags=["researchs"],
    responses={404: {"description": "Not found"},
               200: {"description": "Todo OK jose luis"}}
)

@research_controller.get("/")
async def hola():
    return {"message": "Hello World"}