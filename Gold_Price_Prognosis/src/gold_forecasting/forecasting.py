"""Model construction and recursive frozen-origin forecasting."""
from .models import MODELS

def make_model(name, config, device, seed): return MODELS[name](config=config,device=device,seed=seed)

def frozen_origin_forecast(model, training_values, horizon):
    """Forecast recursively without revealing hidden target observations."""
    return model.predict(training_values, horizon)
