import pandas as pd
from entities.league import League
from services.football_fields import FootballFields
from services.football_data_service import FootballDataService

class FootballNewResultService(FootballDataService):

    def _download(self, league: League) -> pd.DataFrame:
        data = pd.read_csv(league.fixtures_url)
        return data
    
    def _get_fixtures(self, matches: pd.DataFrame) -> pd.DataFrame:
        matches = matches[['Date', 'Season', 'Home', 'Away', 'AvgH', 'AvgD', 'AvgA', 'HG', 'AG', 'Res']]
        matches = matches.rename(columns={
            'Home': FootballFields.HOMETEAM.value,
            'Away': FootballFields.AWAYTEAM.value,
            'AvgH': FootballFields.HOMEPERCENT.value,
            'AvgD': FootballFields.DRAWPERCENT.value,
            'AvgA': FootballFields.AWAYPERCENT.value,
            'Res': FootballFields.RESULT.value,
        })