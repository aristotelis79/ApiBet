import pandas as pd
import numpy as np
import tensorflow.keras.utils as utils
from services.football.football_fields import FootballField, Result

def _columnus():
    return [FootballField.SEASON.value, FootballField.DATE.value, FootballField.RESULT.value,
        FootballField.HOMETEAM.value, FootballField.AWAYTEAM.value, FootballField.HOMEGOAL.value, 
        FootballField.AWAYGOAL.value]

def preprocess_training_dataframe(matches_df: pd.DataFrame, one_hot: bool) -> (np.ndarray, np.ndarray):
    inputs = matches_df.dropna().drop(columns=_columnus())
    inputs = inputs.to_numpy(dtype=np.float64)
    targets = matches_df[FootballField.RESULT.value].replace({
            Result.HOMEWIN.value : 0,
            Result.DRAW.value : 1,
            Result.AWAYWIN.value : 2}).to_numpy(dtype=np.int64)

    if one_hot:
        targets = utils.to_categorical(targets)

    return inputs, targets

def construct_input_from_team_names(
        matches_df: pd.DataFrame,
        home_team: str,
        away_team: str,
        home_odd: float,
        draw_odd: float,
        away_odd: float) -> np.ndarray:
    home_team_row = matches_df[matches_df[FootballField.HOMETEAM.value] == home_team].head(1).drop(columns=_columnus())
    away_team_row = matches_df[matches_df[FootballField.AWAYTEAM.value] == away_team].head(1).drop(columns=_columnus())
    return np.hstack((
        np.float64([home_odd, draw_odd, away_odd]),
        home_team_row[[col for col in home_team_row.columns if col[0] == Result.HOMEWIN.value ]].to_numpy(dtype=np.float64).flatten(),
        away_team_row[[col for col in home_team_row.columns if col[0] == Result.AWAYWIN.value]].to_numpy(dtype=np.float64).flatten()
    )).reshape((1, -1))

def construct_inputs_from_fixtures(
        matches_df: pd.DataFrame,
        fixtures_df: pd.DataFrame) -> np.ndarray:
    return np.vstack([
        construct_input_from_team_names(
            matches_df=matches_df,
            home_team=match[FootballField.HOMETEAM.value],
            away_team=match[FootballField.AWAYTEAM.value],
            odd_1=match[FootballField.HOMEPERCENT.value],
            odd_x=match[FootballField.DRAWPERCENT.value],
            odd_2=match[FootballField.AWAYPERCENT.value])
        for _, match in fixtures_df.iterrows()])

def split_train_targets(
        inputs: np.ndarray,
        targets: np.ndarray,
        num_eval_samples: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x_train = inputs[num_eval_samples:]
    y_train = targets[num_eval_samples:]
    x_test = inputs[: num_eval_samples]
    y_test = targets[: num_eval_samples]
    return x_train, y_train, x_test, y_test

def get_ensemble_predictions(x: np.ndarray, models: list) -> (np.ndarray, np.ndarray):
    sum_predict_proba = np.zeros(shape=(x.shape[0], 3), dtype=np.float64)

    for model in models:
        _, predict_proba = model.predict(x=x)
        sum_predict_proba += predict_proba
    y_prob = sum_predict_proba / len(models)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob