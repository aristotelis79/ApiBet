from fastapi.responses import JSONResponse
import constants
import json
from models.model import Model
from repository.model_repository import ModelRepository
from services.football.football_service import FootballService
from fastapi import Depends, FastAPI
from services.train.tunning_nn_service import TuningNNService
from preprocessing.training import preprocess_training_dataframe

api = FastAPI()

@api.get("/leagues/{country}/{division}")
def leagues(
    country: str,
    division: str,
    footbal_service: FootballService = Depends()):
    matches = footbal_service.get_or_create_league(country, division)
    if matches is None:
        return
    return json.loads(matches.to_json())

@api.put("/leagues/{country}/{division}")
def leagues_update(
    country: str, 
    division: str,
    footbal_service: FootballService = Depends()):
    matches = footbal_service.update_league(country=country, division=division)
    if matches is None:
        return
    return json.loads(matches.to_json())

@api.post("/train/{model_name}/{metric}/{target}/{country}/{division}")
def train(
    model_name: str, 
    metric: str, 
    target: str,
    country: str, 
    division: str, 
    update_league: bool = False, 
    footbal_service: FootballService = Depends(),
    model_repository: ModelRepository = Depends()):
    matches = footbal_service.update_league(country=country, division=division
                                            ) if update_league else footbal_service.get_or_create_league(country, division)
    if matches is None:
        return
    
    match model_name:
        case constants.NN_MODEL_NAME :
            model, model_eval = TuningNNService(random_seed=0,matches_df=matches).train(metric_name=metric,metric_target=target)
            model_repository.store_model(model=model,league_country=country,league_name=division)
        case constants.RF_MODEL_NAME :
            raise NotImplementedError(f'Model "{model_name}" has not been implemented yet')
        case _ :
            raise NotImplementedError(f'Model "{model_name}" has not been implemented yet')
    
    return model_eval

@api.get("/model/{model_name}/{country}/{division}")
def model(
    model_name: str,
    country: str, 
    division: str, 
    one_hot: bool = False,
    footbal_service: FootballService = Depends(),
    model_repository: ModelRepository = Depends()):
    matches = footbal_service.get_or_create_league(country, division)

    if matches is None:
        return
    
    input, _ = preprocess_training_dataframe(matches_df=matches, one_hot=one_hot)

    model = model_repository.load_model(league_country=country, league_name=division, model_name=model_name,input_shape=input.shape[1:])

    x, y, metrics = model.evaluate(matches_df=matches)

    return metrics

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('bet_api:api', host="0.0.0.0", port=5678, reload=True)