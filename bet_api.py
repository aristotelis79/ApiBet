from repository.league_repository import LeagueRepository
from services.football_data_service import FootballDataService
from services.football_result_service import FootballResultService
from entities.league import League
from fastapi import Depends, FastAPI
import json

from services.football_upcoming_result_service import FootballNewResultService

api = FastAPI()

@api.get("/leagues")
def leagues(country: str, division: str, repository: LeagueRepository = Depends()):
    league : League = repository.get_all_available_leagues()[(country,division)]

    if league.league_type == 'main':
        matches = FootballResultService().download(league=league)
    elif league.league_type == 'new':
        matches = FootballNewResultService().download(league=league)

    return json.loads(matches.to_json())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('bet_api:api', host="0.0.0.0", port=5678, reload=True)