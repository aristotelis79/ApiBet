import constants
import pandas as pd
from models.model import Model
from typing import Callable
from services.train.train_service import TrainService
from train_models.tf.nn import FCNet
from tuners.tf.nn import FCNetTuner
from tuners.tuner import Tuner

class TuningNNService(TrainService):
    def __init__(
            self,
            random_seed: int,
            matches_df: pd.DataFrame):
        super().__init__(
            random_seed=random_seed,
            matches_df=matches_df,
            one_hot=True)
        self._epochs_var = 80
        self._early_stopping_epochs_var = 35
        self._learning_rate_decay_factor_var = '0.2'
        self._learning_rate_decay_epochs_var = 10
        self._min_layers_var = 3
        self._max_layers_var = 5
        self._min_units_var = 32
        self._max_units_var = 128
        self._units_increment_var = 16
        self._text = None

    def _construct_tuner(
            self,
            n_trials: int,
            metric: Callable,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            random_seed: int = 0) -> Tuner:
        return FCNetTuner(
            n_trials=n_trials,
            metric=metric,
            matches_df=matches_df,
            num_eval_samples=num_eval_samples,
            epochs=self._epochs_var,
            early_stopping_epochs=self._early_stopping_epochs_var,
            learning_rate_decay_factor=float(self._learning_rate_decay_factor_var),
            learning_rate_decay_epochs=self._learning_rate_decay_epochs_var,
            min_layers=self._min_layers_var,
            max_layers=self._max_layers_var,
            min_units=self._min_units_var,
            max_units=self._max_units_var,
            units_increment=self._units_increment_var,
            random_seed=random_seed)

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return FCNet(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self, model: Model, best_params: dict):
        num_hidden_layers = best_params[constants.NUM_HIDDEN_LAYERS]
        hidden_layers = [best_params[f'layer_{i}'] for i in range(num_hidden_layers)]
        activations = [best_params[f'activation_{i}'] for i in range(num_hidden_layers)]
        batch_normalizations = [best_params[f'bn_{i}'] for i in range(num_hidden_layers)]
        regularizations = [best_params[f'regularization_{i}'] for i in range(num_hidden_layers)]
        dropouts = [best_params[f'dropout_{i}'] for i in range(num_hidden_layers)]

        model.build_model(
            epochs=self._epochs_var,
            batch_size=best_params[constants.BATCH_SIZE],
            early_stopping_epochs=self._early_stopping_epochs_var,
            learning_rate_decay_factor=float(self._learning_rate_decay_factor_var),
            learning_rate_decay_epochs=self._learning_rate_decay_epochs_var,
            learning_rate=best_params[constants.LEARNING_RATE],
            noise_range=best_params[constants.NOISE_RANGE],
            hidden_layers=hidden_layers,
            batch_normalizations=batch_normalizations,
            activations=activations,
            regularizations=regularizations,
            dropouts=dropouts,
            optimizer=best_params[constants.OPTIMIZER])
