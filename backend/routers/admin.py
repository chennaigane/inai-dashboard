from fastapi import APIRouter
router = APIRouter()

@router.get("/roles")
def roles():
    return ["admin","editor","viewer"]
