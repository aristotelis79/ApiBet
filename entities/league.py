class League:

    def  __init__(
            self,
            country: str,
            name: str,
            fixtures_url: str,
            year_start: int,
            league_type: str,
            upcoming_fixtures_url: str
            ) -> None:
        self._country = country
        self._name = name
        self._fixtures_url = fixtures_url
        self._year_start = year_start
        self._league_type = league_type
        self._upcoming_fixtures_url = upcoming_fixtures_url

    @property
    def country(self) -> str:
        return self._country
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def fixtures_url(self) -> str:
        return self._fixtures_url
    
    @property
    def year_start(self) -> str:
        return self._year_start
    
    @year_start.setter
    def year_start(self, year_start: int):
        self._year_start = year_start
    
    @property
    def league_type(self) -> str:
        return self._league_type
    
    @property
    def upcoming_fixtures_url(self) -> str:
        return self._upcoming_fixtures_url
