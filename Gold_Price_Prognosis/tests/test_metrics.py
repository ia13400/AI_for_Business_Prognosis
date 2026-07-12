import pytest
from gold_forecasting.metrics import metric_set
def test_perfect_metrics():
    result=metric_set([1,2,3],[1,2,3],[0,1,2]); assert result["mae"]==0; assert result["rmse"]==0; assert result["smape"]==0; assert result["directional_accuracy"]==1
def test_known_mae(): assert metric_set([1,3],[2,1],[0,1])["mae"]==pytest.approx(1.5)
