import numpy as np
from abc import ABC, abstractmethod
from imblearn.over_sampling import SVMSMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from models.metric import Metric

from services.football_fields import Result


class Model(ABC):
    def __init__(self, input_shape: tuple, random_seed: int):
        self._input_shape = input_shape
        self._random_seed = random_seed
        self._model = None

    ResultsArray = [Result.HOMEWIN.value, Result.DRAW.value, Result.AWAYWIN.value]

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @property
    def model(self):
        return self._model

    @property
    def random_seed(self) -> int:
        return self._random_seed

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def build_model(self, **kwargs):
        self._model = self._build_model(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        pass

    @abstractmethod
    def save(self, checkpoint_filepath: str):
        pass

    def load(self, checkpoint_filepath: str):
        self._model = self._load(checkpoint_filepath=checkpoint_filepath)

    @abstractmethod
    def _load(self, checkpoint_filepath: str):
        pass

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            use_over_sampling: bool
    ) -> dict:
        if use_over_sampling:
            x_train, y_train = SVMSMOTE(
                n_jobs=-1,
                random_state=self.random_seed,
                svm_estimator=SVC(random_state=self.random_seed)
            ).fit_resample(x_train, y_train)

        self._train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        return self.evaluate(x_test=x_test, y_true=y_test)

    @abstractmethod
    def _train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        pass

    def evaluate(self, x_test: np.ndarray, y_true: np.ndarray) -> dict:
        y_actual = np.argmax(y_true, axis=1) if y_true.ndim != 1 else y_true
        y_pred, _ = self.predict(x=x_test)

        f1 = self.metric(f1_score(y_true=y_actual, y_pred=y_pred, average=None))
        precision = self.metric(precision_score(y_true=y_actual, y_pred=y_pred, average=None))
        recall = self.metric(recall_score(y_true=y_actual, y_pred=y_pred, average=None))

        return {
                Metric.ACCURACY: round(accuracy_score(y_true=y_actual, y_pred=y_pred)*100, 2), 
                Metric.F1: f1,
                Metric.PRECISION: precision,
                Metric.RECALL: recall
            }

    def metric(self, metric_val):
        return {
            target: round(score*100, 2)
            for target, score in zip(self.ResultsArray, metric_val)
        }
