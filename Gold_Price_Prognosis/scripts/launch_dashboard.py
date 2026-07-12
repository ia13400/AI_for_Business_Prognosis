"""Launch the artifact-only Streamlit dashboard."""
import subprocess,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gold_forecasting.paths import ROOT
subprocess.run([sys.executable,"-m","streamlit","run",str(ROOT/"dashboard"/"app.py")],check=True)
