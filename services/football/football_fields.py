from enum import Enum

class FootballField(Enum):
    HOMETEAM = 'Home Team'
    AWAYTEAM = 'Away Team'
    HOMEPERCENT = '1'
    DRAWPERCENT = 'X'
    AWAYPERCENT = '2'
    HOMEGOAL = 'HG'
    AWAYGOAL = 'AG'
    RESULT = 'Result'
    SEASON = 'Season'
    DATE = 'Date'
    LAST_N_HOME_WINS = 'HW'
    LAST_N_HOME_LOSSES = 'HL'
    LAST_N_HOME_GOALS = 'HGF'
    LAST_N_HOME_GOALS_AGAINST = 'HGA'
    LAST_N_HOME_GOALS_DIFF = 'HGDW'
    LAST_N_HOME_LOSSES_GOALS_DIFF = 'HGDL'
    LAST_N_HOME_WINS_RATE = 'HW%'
    LAST_N_HOME_DRAW_RATE = 'HD%'
    LAST_N_AWAY_WINS = 'AW'
    LAST_N_AWAY_LOSSES = 'AL'
    LAST_N_AWAY_GOALS = 'AGF'
    LAST_N_AWAY_GOALS_AGAINST = 'AGA'
    LAST_N_AWAY_GOALS_DIFF = 'AGDW'
    LAST_N_AWAY_LOSSES_GOALS_DIFF = 'AGDL'
    LAST_N_AWAY_WINS_RATE = 'AW%'
    LAST_N_AWAY_DRAW_RATE = 'AD%'

class Result(Enum):
    HOMEWIN = 'H'
    DRAW = 'D'
    AWAYWIN = 'A'
    ALL = [HOMEWIN, DRAW, AWAYWIN]