
import pandas as pd
from repository.league_repository import LeagueRepository
from services.football_new_result_service import FootballNewResultService
from services.football_result_service import FootballResultService

class FootballService:

    def __init__(self):
        self._leagueRepository = LeagueRepository()

    def get_league_matches(self, country: str, division: str) -> pd.DataFrame or None:
        league = self._leagueRepository.get_league(country=country, division=division)

        if league is None:
            return None

        matches = self._leagueRepository.load_league(league=league)

        if matches is not None:
            self._leagueRepository.to_csv(df = matches,league=league)
            return matches

        if league.league_type == 'main':
            matches = FootballResultService().download(league=league)
        elif league.league_type == 'new':
            matches = FootballNewResultService().download(league=league)

        self._leagueRepository.save_league(df=matches, league=league)

        return matches