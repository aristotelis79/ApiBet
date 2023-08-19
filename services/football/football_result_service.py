
import pandas as pd
from datetime import date
from entities.league import League
from services.football.football_fields import FootballField
from services.football.football_data_service import FootballDataService

class FootballResultService(FootballDataService):
    
    def _download(self, league: League) -> pd.DataFrame:
        url_list = self._generate_url_list(league = league)
        matches = []
        for i, url in enumerate(url_list):
            try:
                mathes_df = pd.read_csv(url)
                mathes_df[FootballField.SEASON.value] = league.year_start + i
                matches.append(mathes_df)
            except:
                break
        return matches[0] if len(matches) == 1 else pd.concat(matches)
    
    def _get_fixtures(self, mathes_df: pd.DataFrame) -> pd.DataFrame:
        mathes_df = mathes_df[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTHG', 'FTAG', 'FTR']]
        mathes_df = mathes_df.rename(columns={
            'HomeTeam': FootballField.HOMETEAM.value,
            'AwayTeam': FootballField.AWAYTEAM.value,
            'B365H': FootballField.HOMEPERCENT.value,
            'B365D': FootballField.DRAWPERCENT.value,
            'B365A': FootballField.AWAYPERCENT.value,
            'FTHG': FootballField.HOMEGOAL.value,
            'FTAG': FootballField.AWAYGOAL.value,
            'FTR': FootballField.RESULT.value})
        return mathes_df

    def _generate_url_list(self, league: League) -> list[str]:
        return [league.fixtures_url.format(self._get_season_years_pattern(year))
                for year in range(league.year_start, date.today().year + 1)]
    
    @staticmethod
    def _get_season_years_pattern(year:int) -> str:
        return f'{str(year)[2:]}{str(year+1)[2:]}'