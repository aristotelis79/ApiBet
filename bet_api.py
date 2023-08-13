import json
from services.football_service import FootballService
from fastapi import Depends, FastAPI

api = FastAPI()

@api.get("/leagues")
def leagues(country: str, division: str, footbal_service: FootballService = Depends()):
    matches = footbal_service.get_or_create_league(country, division)
    return json.loads(matches.to_json())

@api.get("/leagues/update")
def leagues_update(country: str, division: str, footbal_service: FootballService = Depends()):
    matches = footbal_service.update_league(country=country, division=division)
    return json.loads(matches.to_json())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('bet_api:api', host="0.0.0.0", port=5678, reload=True)