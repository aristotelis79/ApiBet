from pandas import DataFrame
from abc import ABC, abstractmethod
from entities.league import League

class FootballDataService(ABC):
       
    def download(self, league:League) -> DataFrame:
        matches = self._download(league=league)
        matches = self._get_fixtures(matches)
        matches = matches.drop_duplicates()
        matches = matches.iloc[::-1].reset_index(drop=True)
        return matches

    @abstractmethod
    def _download(self, league: League) -> DataFrame:
        pass

    @abstractmethod
    def _get_fixtures(self, matches: DataFrame) -> DataFrame:
        pass