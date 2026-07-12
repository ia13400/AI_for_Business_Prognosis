"""Launch the local MLflow UI."""
import subprocess,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gold_forecasting.paths import MLFLOW
subprocess.run([sys.executable,"-m","mlflow","ui","--backend-store-uri",MLFLOW.resolve().as_uri()],check=True)
