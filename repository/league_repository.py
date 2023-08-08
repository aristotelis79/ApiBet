import os
import constants
import csv
from entities.league import League

class LeagueRepository:

    def __init__(self) -> None:
        self._available_leagues_filepath = constants.AVAILABLE_LEAGUES_FILEPATH
        self._saved_leagues_directory = constants.SAVED_LEAGUES_DIRECTORY
        os.makedirs(self._saved_leagues_directory, exist_ok=True)

    def get_all_available_leagues(self) -> dict:
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