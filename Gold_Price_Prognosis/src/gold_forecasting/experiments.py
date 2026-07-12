"""End-to-end holdout and future experiment orchestration."""
from datetime import datetime, timezone
import json, time
import mlflow
import pandas as pd
from .cache import ArtifactCache
from .config import load_yaml, select_device, set_seed
from .forecasting import make_model, frozen_origin_forecast
from .hashing import dataframe_hash, stable_hash
from .metrics import horizon_metrics
from .paths import CACHE,CHECKPOINTS,PREDICTIONS,METRICS,ensure_directories
from .plotting import plot_predictions,plot_residuals,plot_losses,plot_combined,plot_split
from .splitting import historical_split
from .mlflow_utils import tracked_run,log_dict_flat
from .progress import progress

def _configs(mode=None):
    models=load_yaml("models.yaml"); experiments=load_yaml("experiments.yaml"); mode=mode or experiments["mode"]
    return models,experiments,mode
def _model_config(models,name,mode):
    config=dict(models.get(name,{})); overrides=models.get(mode,{})
    for key,value in overrides.items():
        if key in config: config[key]=value
    return config
def _run_model(name,train,validation,horizon,namespace,data_hash,models,mode,seed,force_retrain,meta):
    config=_model_config(models,name,mode); inputs={"pipeline":"1.0.0","namespace":namespace,"data_hash":data_hash,"model":name,"hyperparameters":config,"seed":seed,"horizon":horizon,**meta}; signature=stable_hash(inputs)
    prediction_path=PREDICTIONS/namespace/f"{name}_{signature[:12]}.csv"; metric_path=METRICS/namespace/f"{name}_{signature[:12]}.csv"; checkpoint=CHECKPOINTS/namespace/f"{name}_{signature[:12]}.pt"; manifest=prediction_path.with_suffix(".manifest.json")
    prediction_path.parent.mkdir(parents=True,exist_ok=True); metric_path.parent.mkdir(parents=True,exist_ok=True)
    cache=ArtifactCache(CACHE); cached=prediction_path.exists() and manifest.exists() and not force_retrain
    if cached: return pd.read_csv(prediction_path,parse_dates=["date"],index_col="date"),signature
    started=time.perf_counter(); model=make_model(name,config,select_device(),seed); model.fit(train,validation,checkpoint if name not in ("naive","arima") else None); predicted=frozen_origin_forecast(model,train,horizon)
    frame=pd.DataFrame({"predicted":predicted}); frame.to_csv(prediction_path,index=False)
    tags={"data_hash":data_hash,"cache_signature":signature,**meta}
    with tracked_run(name,namespace,tags) as run:
        log_dict_flat("model",config); mlflow.log_param("seed",seed); mlflow.log_param("device",str(select_device())); mlflow.log_metric("runtime_seconds",time.perf_counter()-started); mlflow.log_artifact(str(prediction_path));
        if checkpoint.exists(): mlflow.log_artifact(str(checkpoint))
        inputs["mlflow_run_id"]=run.info.run_id
    cache.write_manifest(manifest,signature,inputs)
    return frame,signature

def run_holdout(frame, cutoff="2025-06-01", mode=None, force_retrain=False):
    ensure_directories(); models,exp,mode=_configs(mode); seed=int(models["seed"]); set_seed(seed); split=historical_split(frame,cutoff,exp["validation_fraction"]); target="gold_usd"; data_hash=dataframe_hash(frame); outputs={}; combined=pd.DataFrame({"actual":split.test[target]})
    plot_split(frame[target],cutoff,data_hash)
    for name in progress(["naive","arima","lstm","nbeats","patchtst"],"Holdout models"):
        raw,signature=_run_model(name,split.train[target].values,split.validation[target].values,len(split.test),"holdout",data_hash,models,mode,seed,force_retrain,{"cutoff":cutoff,"train_range":f"{split.train.index.min()}:{split.train.index.max()}","validation_range":f"{split.validation.index.min()}:{split.validation.index.max()}","test_range":f"{split.test.index.min()}:{split.test.index.max()}"})
        raw.index=split.test.index; raw.index.name="date"; result=pd.DataFrame({"actual":split.test[target],"predicted":raw.predicted}); result.to_csv(PREDICTIONS/"holdout"/f"{name}_{signature[:12]}.csv"); metrics=horizon_metrics(result,split.train[target],exp["holdout_horizons"]); metrics.insert(0,"model",name); metrics.to_csv(METRICS/"holdout"/f"{name}_{signature[:12]}.csv",index=False); outputs[name]=result; combined[name]=result.predicted; plot_predictions(result,"holdout",name,signature,cutoff); plot_residuals(result,name,signature)
    plot_combined(combined,"holdout",data_hash); return outputs

def run_future(frame,horizon=90,mode=None,force_retrain=False):
    ensure_directories(); models,exp,mode=_configs(mode); seed=int(models["seed"]); set_seed(seed); target=frame.gold_usd.dropna(); validation_size=max(1,int(len(target)*exp["validation_fraction"])); train,validation=target.iloc[:-validation_size],target.iloc[-validation_size:]; data_hash=dataframe_hash(frame); dates=pd.bdate_range(target.index.max()+pd.offsets.BDay(),periods=horizon); combined=pd.DataFrame(index=dates)
    for name in progress(["naive","arima","lstm","nbeats","patchtst"],"Future models"):
        raw,signature=_run_model(name,train.values,validation.values,horizon,"future",data_hash,models,mode,seed,force_retrain,{"forecast_origin":str(target.index.max()),"approach":"univariate-no-future-exogenous"}); raw.index=dates; raw.index.name="date"; raw.to_csv(PREDICTIONS/"future"/f"{name}_{signature[:12]}.csv"); combined[name]=raw.predicted; plot_predictions(raw,"future",name,signature)
    combined.to_csv(PREDICTIONS/"future"/f"all_models_{data_hash[:12]}.csv"); plot_combined(combined,"future",data_hash); return combined
