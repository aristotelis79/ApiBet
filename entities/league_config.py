from preprocessing.statistics import StatisticsEngine

class LeagueConfig:
    def  __init__(
            self,
            country: str,
            name: str,
            last_n_matches: int = 5,
            goal_diff_margin: int = 3
            ) -> None:
        self._country = country
        self._name = name
        self._last_n_matches = last_n_matches
        self._goal_diff_margin = goal_diff_margin
        self._statistic_columns = StatisticsEngine.Columns

    @property
    def country(self) -> str:
        return self._country
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def last_n_matches(self) -> int:
        return self._last_n_matches
    
    @property
    def goal_diff_margin(self) -> int:
        return self._goal_diff_margin
    
    @property
    def statistic_columns(self) -> list[str]:
        return self._statistic_columns