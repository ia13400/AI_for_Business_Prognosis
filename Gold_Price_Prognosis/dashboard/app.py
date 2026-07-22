"""German artifact-only dashboard; it never trains models."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gold_forecasting.paths import PROCESSED,PREDICTIONS,METRICS

st.set_page_config(page_title="Goldpreisprognose",page_icon="📈",layout="wide")
st.title("Goldpreisprognose")
st.warning("Modellbasierte Prognosen sind unsicher und keine Anlageberatung.")
tabs=st.tabs(["Überblick","Datenexploration","Univariable","Multivariant Forecasting","MLflow"])
datasets=sorted(PROCESSED.glob("gold_dataset_*.parquet"),key=lambda p:p.stat().st_mtime,reverse=True)
data=pd.read_parquet(datasets[0]) if datasets else None

def _experiment_tab(namespace: str, description: str, future_enabled: bool):
    prediction_dir=PREDICTIONS/namespace; metric_dir=METRICS/namespace
    files=sorted(prediction_dir.glob("*.csv")) if prediction_dir.exists() else []
    st.caption(description)
    if not files:
        st.info(f"Bitte zuerst das Experiment ausführen (Notebook, Kapitel 4: {namespace}).")
        return
    metric_files=list(metric_dir.glob("*.csv")) if metric_dir.exists() else []
    metrics=pd.concat([pd.read_csv(p) for p in metric_files],ignore_index=True) if metric_files else pd.DataFrame()
    st.subheader("Modellvergleich (Testzeitraum)")
    st.dataframe(metrics.sort_values(["horizon","mae"]) if not metrics.empty else metrics)
    selected=st.selectbox("Modell",files,format_func=lambda p:p.stem,key=f"{namespace}_model")
    prediction=pd.read_csv(selected,parse_dates=["date"]).set_index("date")[["actual","predicted"]]
    st.line_chart(prediction)
    st.download_button("Prognose herunterladen",prediction.to_csv().encode(),selected.name,"text/csv",key=f"{namespace}_download")
    if "actual" in prediction.columns and "predicted" in prediction.columns:
        st.line_chart((prediction["actual"]-prediction["predicted"]).rename("Residuum"))
    if future_enabled:
        st.subheader("Echte Zukunftsprognose (nur univariate Modelle)")
        future_files=sorted((PREDICTIONS/"future").glob("*.csv")) if (PREDICTIONS/"future").exists() else []
        if not future_files:
            st.info("Bitte zuerst die Zukunftsprognose ausführen (Notebook, Kapitel 6).")
        else:
            future_model=st.selectbox("Modell (Zukunft)",future_files,format_func=lambda p:p.stem,key="future_model")
            future=pd.read_csv(future_model,parse_dates=["date"],index_col="date")
            horizon=st.select_slider("Handelstage",options=[7,30,90],value=30,key="future_horizon")
            st.line_chart(future.iloc[:horizon])
            st.download_button("Zukunftsprognose herunterladen",future.iloc[:horizon].to_csv().encode(),"zukunftsprognose.csv","text/csv",key="future_download")
            st.caption("Univariate Methode ohne unbekannte zukünftige exogene Variablen -- die einzige hier zulässige echte Zukunftsprognose.")

with tabs[0]:
    st.subheader("Projektüberblick")
    st.write("Zielvariable: täglicher Goldpreis in USD je Feinunze. Experiment 1 (univariat, ohne exogene Variablen): SARIMA, PatchTST, Chronos (Original & Bolt, Zero-Shot). Experiment 2 (multivariat, mit exogenen Variablen): SARIMAX, XGBoost, TFT (nativ, kompakt).")
    st.write("Validierungs- und Testzeitraum sind je 1 Jahr lang; die Auswertung erfolgt rollierend (walk-forward, Standard: 20 Tage Horizont, 20 Tage Schritt). Alle Werte sind in `configs/experiments.yaml` (`split`, `rolling`) konfigurierbar; das Enddatum der heruntergeladenen Daten steht in `configs/data.yaml`.")
    if data is None: st.info("Bitte zuerst Daten herunterladen und Experimente ausführen.")
    else:
        c1,c2,c3=st.columns(3); c1.metric("Beginn",str(data.index.min().date())); c2.metric("Ende",str(data.index.max().date())); c3.metric("Letzter Preis",f"{data.gold_usd.iloc[-1]:,.2f} USD")
with tabs[1]:
    st.subheader("Datenexploration")
    if data is not None:
        st.line_chart(data.gold_usd); st.dataframe(pd.DataFrame({"Fehlende Werte":data.isna().sum(),"Abdeckung in %":100*(1-data.isna().mean())})); st.dataframe(data.corr(numeric_only=True).style.background_gradient(cmap="RdBu",vmin=-1,vmax=1))
    else: st.info("Keine aufbereiteten Daten gefunden.")
with tabs[2]:
    _experiment_tab("univariate","Experiment 1: Prognose ohne exogene Variablen (SARIMA, PatchTST, Chronos) gegen Naiv und gleitenden Durchschnitt.",future_enabled=True)
with tabs[3]:
    _experiment_tab("multivariate","Experiment 2: Prognose mit exogenen Variablen aus data.yaml (SARIMAX, XGBoost, TFT). Rückblickender Test mit tatsächlich realisierten exogenen Werten (kein unbekannt-zukünftiges Szenario).",future_enabled=False)
with tabs[4]: st.write("Lokale Oberfläche starten: `uv run python scripts/launch_mlflow.py`. Tracking-Daten liegen unter `artifacts/mlflow/`.")
