import numpy as np
import pytest
from gold_forecasting.config import select_device,set_seed
from gold_forecasting.models import NaiveForecaster,ArimaForecaster,LSTMForecaster,NBeatsForecaster,PatchTSTForecaster

SERIES=100+np.sin(np.arange(100)/5)+np.arange(100)*.1
@pytest.mark.parametrize("model,config",[(NaiveForecaster,{}),(ArimaForecaster,{"order":[1,1,0]}),(LSTMForecaster,{"context_length":16,"hidden_size":8,"num_layers":1,"dropout":0,"learning_rate":.01,"batch_size":16,"epochs":1,"patience":1}),(NBeatsForecaster,{"context_length":16,"hidden_size":8,"blocks":1,"learning_rate":.01,"batch_size":16,"epochs":1,"patience":1}),(PatchTSTForecaster,{"context_length":16,"patch_length":4,"stride":4,"d_model":8,"nhead":2,"layers":1,"learning_rate":.01,"batch_size":16,"epochs":1,"patience":1})])
def test_model_fit_predict(model,config,tmp_path):
    set_seed(42); instance=model(config=config,device=select_device(),seed=42); instance.fit(SERIES[:80],SERIES[80:90],tmp_path/"model.pt"); prediction=instance.predict(SERIES[:80],3); assert prediction.shape==(3,) and np.isfinite(prediction).all()
