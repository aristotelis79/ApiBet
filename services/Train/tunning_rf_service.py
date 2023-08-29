import constants
import pandas as pd
from typing import Callable
from models.model import Model
from services.train.train_service import TrainService
from train_models.scikit.rf import RandomForest
from tuners.scikit.rf import RandomForestTuner
from tuners.tuner import Tuner

class TunningRFService(TrainService):
    def __init__(
            self,
            random_seed: int,
            matches_df: pd.DataFrame,
            one_hot:bool = False) -> None:
        super().__init__(
            random_seed=random_seed,
            matches_df=matches_df,
            one_hot=one_hot)
        self._one_hot = one_hot
        
    def _construct_tuner(
            self,
            n_trials: int,
            metric: Callable,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            random_seed: int = 0) -> Tuner:
        return RandomForestTuner(
            n_trials=n_trials,
            metric=metric,
            matches_df=matches_df,
            one_hot=self._one_hot,
            num_eval_samples=num_eval_samples,
            random_seed=random_seed)

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return RandomForest(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self, model: Model, best_params: dict):
        model.build_model(
            n_estimators=best_params[constants.N_ESTIMATORS],
            max_features=best_params[constants.MAX_FEATURES],
            max_depth=best_params[constants.MAX_DEPTH],
            min_samples_leaf=best_params[constants.MIN_SAMPLES_LEAF],
            min_samples_split=best_params[constants.MIN_SAMPLES_SPLIT],
            bootstrap=best_params[constants.BOOTSTRAP],
            class_weight=best_params[constants.CLASS_WEIGHT],
            is_calibrated=best_params[constants.IS_CALIBRATED])