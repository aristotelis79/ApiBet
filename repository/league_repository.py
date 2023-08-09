import os
import constants
import csv
import json
import encodings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union
from entities.league import League
from entities.league_config import LeagueConfig
from preprocessing.statistics import StatisticsEngine

class LeagueRepository:

    def __init__(self):
        self._available_leagues_filepath = constants.AVAILABLE_LEAGUES_FILEPATH
        self._saved_leagues_directory = constants.SAVED_LEAGUES_DIRECTORY
        os.makedirs(self._saved_leagues_directory, exist_ok=True)

    def get_league(self, country: str, division: str) -> League or None:
         return self._get_all_available_leagues()[(country,division)]

    def create_league(self, league: League, league_config: LeagueConfig) -> None:
        matches_df = self.load_league(league)

        if matches_df is None:
            return None

        return self.save_league(
            matches_df,
            league,
            league_config)
    
    def load_league(self, league: League) -> Union[pd.DataFrame, None]:
        if self._league_exists(league.country, league.name):
            league_table = pq.read_table(self._league_table_path(league.country, league.name))
            return league_table.to_pandas()
        return None

    def save_league(self, 
                    df: pd.DataFrame,
                    league: League, 
                    league_config: LeagueConfig, 
                    update_on_exist: bool = True) -> None:
        
        if self._league_exists(league.country, league.name) and update_on_exist is False:
            return None
        
        matches_df = StatisticsEngine(
            matches_df=df,
            last_n_matches=league_config.last_n_matches,
            goal_diff_margin=league_config.goal_diff_margin
        ).compute_statistics(statistic_columns=league_config.statistic_columns)

        table = pa.Table.from_pandas(matches_df)
        pa.parquet.write_table(table, self._league_table_path(league.country, league.name))

        self._save_league_config(league_config)

    def delete_league(self, league: League) -> bool:
        if self._league_exists(league.country, league.name) is False:
            return False
        
        os.remove(self._league_table_path(league.country, league.name))
        os.remove(self._league_config_path(league.country, league.name))
        return True

    def get_league_config(self, country: str, division: str) -> LeagueConfig or None:
        with open(self._league_config_path(country, division), 'r', encoding=encodings.utf_8.getregentry().name) as fp:
            league_config: LeagueConfig = json.load(fp) 
        return league_config

    def to_csv(self, df: pd.DataFrame, league: League) -> None:
        league_filepath = f'{self._saved_leagues_directory}{league.country}{league.name}.csv'
        df.to_csv(league_filepath, index=False)

    def _get_all_available_leagues(self) -> dict:
        with open(file=constants.AVAILABLE_LEAGUES_FILEPATH, 
                  mode='r', encoding= encodings.utf_8.getregentry().name) as csvfile:
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
        return os.path.exists(self._league_table_path(country,divivsion))
    
    def _league_table_path(self, country: str, divivsion: str) -> str:
        return self._saved_leagues_directory + constants.LEAGUE_TABLE.format(country,divivsion)

    def _league_config_path(self, country: str, divivsion: str) -> str:
        return self._saved_leagues_directory + constants.LEAGUE_CONFIG.format(country,divivsion)

    def _save_league_config(self, league_config: LeagueConfig) -> None:
        with open(self._league_config_path(league_config.country, league_config.name),
                   'r', encoding=encodings.utf_8.getregentry().name) as fp:
            json.dump(league_config, fp)