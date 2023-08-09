import os
import constants
import csv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from entities.league import League

class LeagueRepository:

    def __init__(self):
        self._available_leagues_filepath = constants.AVAILABLE_LEAGUES_FILEPATH
        self._saved_leagues_directory = constants.SAVED_LEAGUES_DIRECTORY
        self._league_table_pattern = constants.LEAGUE_TABLE
        os.makedirs(self._saved_leagues_directory, exist_ok=True)

    def get_league(self, country: str, division: str) -> League or None:
         return self._get_all_available_leagues()[(country,division)]
    
    def load_league(self, league: League) -> pd.DataFrame or None:
        if self._league_exists(league.country, league.name):
            league_table = pq.read_table(self._league_table(league.country, league.name))
            return league_table.to_pandas()
        else:
            return None

    def save_league(self, df: pd.DataFrame, league: League) -> None:
        if self._league_exists(league.country, league.name):
            return None
        else:
            table = pa.Table.from_pandas(df)
            pa.parquet.write_table(table, self._league_table(league.country, league.name))

    def _get_all_available_leagues(self) -> dict:
        with open(file=constants.AVAILABLE_LEAGUES_FILEPATH, mode='r', encoding='utf=8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            return {(row[0], row[1]): League(
                country=row[0],
                name=row[1],
                fixtures_url=row[2],
                year_start=int(row[3]),
                league_type=row[4],
                upcoming_fixtures_url=row[5]) for row in reader}

    def _league_exists(self, country: str, divivsion: str) -> bool:
        return os.path.exists(self._league_table(country,divivsion))

    def _league_table(self, country: str, divivsion: str) -> str:
        return self._saved_leagues_directory + self._league_table_pattern.format(country,divivsion)