import constants
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from models.model import Model
from typing import Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models.metric import Metric
from tuners.tuner import Tuner

class TrainService(ABC):
    def __init__(
            self,
            random_seed: int,
            matches_df: pd.DataFrame,
            one_hot: bool) -> None:
        self._random_seed = random_seed
        self._matches_df = matches_df
        self._one_hot = one_hot
        self._metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
        self._metric_targets = {'Home': 0, 'Draw': 1, 'Away': 2}
        self._best_params = None
        self._eval_metrics = None
        self._n_trials_var = 100
        self._metric_var = 'Accuracy'
        self._metric_target_var = 'Home'
        self._num_eval_samples_var = 50

    @property
    def n_trials_var(self) -> int:
        return self._n_trials_var

    @property
    def metrics(self) -> list:
        return self._metrics

    @property
    def metric_targets(self) -> dict:
        return self._metric_targets

    @property
    def metric_var(self) -> str:
        return self._metric_var

    @property
    def metric_target_var(self) -> str:
        return self._metric_target_var

    @property
    def num_eval_samples_var(self) -> int:
        return self._num_eval_samples_var

    def train(self, metric_name: str, metric_target: str) -> (Model,dict or None):
        match metric_name:
            case Metric.ACCURACY.value:
                metric = lambda y_true, y_pred: accuracy_score(y_true=y_true, y_pred=y_pred)
            case Metric.F1.value:
                metric = lambda y_true, y_pred: f1_score(y_true=y_true, y_pred=y_pred, average=None)[metric_target]
            case Metric.PRECISION.value:
                metric = lambda y_true, y_pred: precision_score(y_true=y_true, y_pred=y_pred, average=None)[metric_target]
            case Metric.RECALL.value:
                metric = lambda y_true, y_pred: recall_score(y_true=y_true, y_pred=y_pred, average=None)[metric_target]
            case _:
                raise NotImplementedError(f'Error: Metric "{metric_name}" has not been implemented yet')
        
        tuner = self._construct_tuner(
            n_trials=self._n_trials_var,
            metric=metric,
            matches_df=self._matches_df,
            num_eval_samples=self._num_eval_samples_var,
            random_seed=self._random_seed)
         
        self._best_params = tuner.tune()
        
        model = self._train(
            x_train=tuner.x_train,
            y_train=tuner.y_train,
            x_test=tuner.x_test,
            y_test=tuner.y_test,
            random_seed=self._random_seed,
            best_params=self._best_params)
        
        return (model, self._eval_metrics)
        
    def _train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            random_seed: int,
            best_params: dict):
        model = self._construct_model(input_shape=x_train.shape[1:], random_seed=random_seed)
        self._build_model(model=model, best_params=best_params)
        self._eval_metrics = model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            use_over_sampling=best_params[constants.USER_OVER_SAMPLING])
        return model

    @abstractmethod
    def _construct_tuner(
            self,
            n_trials: int,
            metric: Callable,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            random_seed: int = 0) -> Tuner:
        pass

    @abstractmethod
    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        pass

    @abstractmethod
    def _build_model(self, model: Model, best_params: dict):
        pass