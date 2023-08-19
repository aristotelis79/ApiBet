import os
import constants
import shutil
from models.model import Model
from train_models.scikit.rf import RandomForest
from train_models.tf.nn import FCNet

class ModelRepository:
    def __init__(self) -> None:
        self._models_checkpoint_directory = constants.MODELS_CHECKPOINT_DIRECTORY
        os.makedirs(name=self._models_checkpoint_directory,exist_ok=True)

    def store_model(self, model: Model, league_country: str, league_name: str):
        model.save(f'{self._checkpoint_directory(league_country, league_name)}{model.get_model_name()}')

    def get_all_models(self, league_country: str, league_name: str) -> list | None:
        checkpoint_directory = self._checkpoint_directory(league_country, league_name)
        if not os.path.exists(checkpoint_directory):
            return None
        
        return [name.split('.')[0] for name in os.listdir(checkpoint_directory)]

    def load_model(
            self,
            league_name: str,
            league_country: str,
            model_name: str,
            input_shape: tuple,
            random_seed: int) -> Model | None:
        checkpoint_filepath = self._checkpoint_train_model_directory(league_country,league_name, model_name)
        if os.path.exists(checkpoint_filepath) is False:
            return None
        
        match model_name:
            case constants.RF_MODEL_NAME:
                model = RandomForest(input_shape=input_shape, random_seed=random_seed)
            case constants.NN_MODEL_NAME:
                model = FCNet(input_shape=input_shape, random_seed=random_seed)
            case _:
                raise NotImplementedError(f'Type of model "{model_name}" has not been implemented yet')

        model.load(checkpoint_filepath=checkpoint_filepath)
        return model

    def delete_model(self, league_country: str, league_name: str, model_name: str) -> bool:
        checkpoint_filepath = self._checkpoint_train_model_directory(league_country,league_name, model_name)

        if os.path.exists(checkpoint_filepath):
            shutil.rmtree(checkpoint_filepath) if os.path.isdir(checkpoint_filepath) else os.remove(checkpoint_filepath)
            return True
        
        return False

    def _checkpoint_directory(self, league_country, league_name):
        return f'{self._models_checkpoint_directory}{league_country}.{league_name}/'
    
    def _checkpoint_train_model_directory(self, league_country, league_name, model_name):
        return f'{self._checkpoint_directory(league_country, league_name)}{model_name}'