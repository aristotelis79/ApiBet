import constants
import json
from repository.model_repository import ModelRepository
from services.football.football_service import FootballService
from fastapi import Depends, FastAPI
from services.train.tunning_nn_service import TuningNNService

api = FastAPI()

@api.get("/leagues")
def leagues(
    country: str,
    division: str,
    footbal_service: FootballService = Depends()):
    matches = footbal_service.get_or_create_league(country, division)
    if matches is None:
        return
    return json.loads(matches.to_json())

@api.get("/leagues/update")
def leagues_update(
    country: str, 
    division: str,
    footbal_service: FootballService = Depends()):
    matches = footbal_service.update_league(country=country, division=division)
    if matches is None:
        return
    return json.loads(matches.to_json())

@api.get("/train/{model}/{metric}/{target}")
def train(
    model: str, 
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
    
    match model:
        case constants.NN_MODEL_NAME :
            train_model = TuningNNService(random_seed=0,matches_df=matches)
            train_model = train_model.train(metric_name=metric,metric_target=target)
            model_repository.store_model(model=train_model,league_country=country,league_name=division)
        case constants.RF_MODEL_NAME :
            raise NotImplementedError(f'Model "{model}" has not been implemented yet')
        case _ :
            raise NotImplementedError(f'Model "{model}" has not been implemented yet')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('bet_api:api', host="0.0.0.0", port=5678, reload=True)