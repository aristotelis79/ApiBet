
import pandas as pd
from datetime import date
from entities.league import League
from services.football_fields import FootballFields
from services.football_data_service import FootballDataService

class FootballResultService(FootballDataService):
    
    def _download(self, league: League) -> pd.DataFrame:
        url_list = self._generate_url_list(league = league)
        matches = []
        for i, url in enumerate(url_list):
            try:
                mathes_df = pd.read_csv(url)
                mathes_df['Season'] = league.year_start + i
                matches.append(mathes_df)
            except:
                break
        return matches[0] if len(matches) == 1 else pd.concat(matches)
    
    def _get_fixtures(self, mathes_df: pd.DataFrame) -> pd.DataFrame:
        mathes_df = mathes_df[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTHG', 'FTAG', 'FTR']]
        mathes_df = mathes_df.rename(columns={
            'HomeTeam': FootballFields.HOMETEAM,
            'AwayTeam': FootballFields.AWAYTEAM,
            'B365H': FootballFields.HOMEPERCENT,
            'B365D': FootballFields.DRAWPERCENT,
            'B365A': FootballFields.AWAYPERCENT,
            'FTHG': FootballFields.HOMEGOAL,
            'FTAG': FootballFields.AWAYGOAL,
            'FTR': FootballFields.RESULT})
        return mathes_df

    def _generate_url_list(self, league: League) -> list[str]:
        return [league.fixtures_url.format(self._get_season_years_pattern(year))
                for year in range(league.year_start, date.today().year + 1)]
    
    @staticmethod
    def _get_season_years_pattern(year:int) -> str:
        return f'{str(year)[2:]}{str(year+1)[2:]}'