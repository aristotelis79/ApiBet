import constants
import json
from models.model import Model
from repository.model_repository import ModelRepository
from services.football.football_fields import Result
from services.football.football_service import FootballService
from fastapi import Depends, FastAPI
from services.train.tunning_nn_service import TuningNNService
from preprocessing.training import preprocess_training_dataframe,construct_input_from_team_names,get_all_predictions,filter_matches_by_odd
from services.train.tunning_rf_service import TunningRFService

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
            model, model_eval = TunningRFService(random_seed=0,matches_df=matches).train(metric_name=metric,metric_target=target)
            model_repository.store_model(model=model,league_country=country,league_name=division)
        case _ :
            raise NotImplementedError(f'Model "{model_name}" has not been implemented yet')
    
    return model_eval

@api.get("/predict/{model_name}/{country}/{division}/{home_team}/{away_team}/{home_odd}/{draw_odd}/{away_odd}")
def predict(
    model_name: str,
    country: str, 
    division: str, 
    home_team: str,
    away_team: str,
    home_odd: float,
    draw_odd: float,
    away_odd: float,
    footbal_service: FootballService = Depends(),
    model_repository: ModelRepository = Depends()):
    matches = footbal_service.get_or_create_league(country, division)

    inputs = construct_input_from_team_names(
        matches_df=matches,
        home_team=home_team,
        away_team=away_team,
        home_odd=home_odd,
        draw_odd=draw_odd,
        away_odd=away_odd
    )

    match model_name:
        case constants.NN_MODEL_NAME | constants.RF_MODEL_NAME :
            model = model_repository.load_model(league_country=country, league_name=division, model_name=model_name,input_shape=inputs.shape[1:])
            y_pred, predict_proba = model.predict(x=inputs)
        case constants.ALL_MODEL_NAME:
            models = [
                model_repository.load_model(league_country=country, league_name=division, model_name=name,input_shape=inputs.shape[1:])
                for name in model_repository.get_all_models(league_country=country, league_name=division)
            ]
            y_pred, predict_proba = get_all_predictions(x=inputs, models=models)
        case _ :
            raise NotImplementedError(f'Model "{model_name}" has not been implemented yet')
        
    return { 
        Result.HOMEWIN.value: str(predict_proba[0][0]),
        Result.DRAW.value: str(predict_proba[0][1]),
        Result.AWAYWIN.value: str(predict_proba[0][2]),
    }

@api.get("/evaluate/{model_name}/{country}/{division}")
def evaluate(    
    model_name: str,
    country: str, 
    division: str,
    sample_number: int = None,
    filter: str = None,
    one_hot: bool = False,
    footbal_service: FootballService = Depends(),
    model_repository: ModelRepository = Depends()):
    matches = footbal_service.get_or_create_league(country, division)
    
    if(sample_number is not None):
        matches = matches.iloc[0: sample_number]
    
    if matches is None:
        return
    
    if(filter is not None):
        matches = filter_matches_by_odd(filter=filter,matches_df=matches)

    inputs, _ = preprocess_training_dataframe(matches_df=matches, one_hot=one_hot)

    match model_name:
        case constants.NN_MODEL_NAME | constants.RF_MODEL_NAME:
            model = model_repository.load_model(league_country=country, league_name=division, model_name=model_name,input_shape=inputs.shape[1:])
            y_pred, predict_proba, metric = model.evaluate_scores(matches_df=matches)
        case constants.ALL_MODEL_NAME:
            models = [
                model_repository.load_model(league_country=country, league_name=division, model_name=name,input_shape=inputs.shape[1:])
                for name in model_repository.get_all_models(league_country=country, league_name=division)
            ]
            metrics = {}
            for model in models:
                y_pred, predict_proba, metric = model.evaluate_scores(matches_df=matches)
                metrics[model.get_model_name()] = metric
        case _ :
            raise NotImplementedError(f'Model "{model_name}" has not been implemented yet')

    return metric

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('bet_api:api', host="0.0.0.0", port=5678, reload=True)