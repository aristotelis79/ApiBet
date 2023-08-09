
from typing import Union
import pandas as pd
from entities.league import League
from entities.league_config import LeagueConfig
from repository.league_repository import LeagueRepository
from services.football_new_result_service import FootballNewResultService
from services.football_result_service import FootballResultService

class FootballService:

    def __init__(self):
        self._leagueRepository = LeagueRepository()

    def get_or_create_league(self, country: str, division: str) -> Union[pd.DataFrame,None]:
        league = self._leagueRepository.get_league(country=country, division=division)

        if league is None:
            return None

        matches = self._leagueRepository.load_league(league=league)

        if matches is not None:
            #self._leagueRepository.to_csv(df = matches,league=league)
            return matches

        if league.league_type == 'main':
            matches = FootballResultService().download(league=league)
        elif league.league_type == 'new':
            matches = FootballNewResultService().download(league=league)

        if matches is None:
            return None

        league_config = LeagueConfig(league.country,league.name)

        self._leagueRepository.save_league(df=matches, league=league, league_config=league_config)

        return matches

    def update_league(self, league: League) -> None:
        league_config = self._leagueRepository.get_league_config(league.country,league.name)

        if league_config is None:
            return None

        if  self._leagueRepository.delete_league(league=league) is False:
            return None            
    
        return  self._leagueRepository.create_league(
            league=league,
            league_config = league_config)