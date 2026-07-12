"""German artifact-only dashboard; it never trains models."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gold_forecasting.paths import PROCESSED,PREDICTIONS,METRICS,FIGURES

st.set_page_config(page_title="Goldpreisprognose",page_icon="📈",layout="wide")
st.title("Goldpreisprognose")
st.warning("Modellbasierte Prognosen sind unsicher und keine Anlageberatung.")
tabs=st.tabs(["Überblick","Datenexploration","Holdout ab Juni 2025","Zukunftsprognosen","MLflow"])
datasets=sorted(PROCESSED.glob("gold_dataset_*.parquet"),key=lambda p:p.stat().st_mtime,reverse=True)
data=pd.read_parquet(datasets[0]) if datasets else None
with tabs[0]:
    st.subheader("Projektüberblick"); st.write("Zielvariable: täglicher Goldpreis in USD je Feinunze. Modelle: Naiv, ARIMA, LSTM, N-BEATS und PatchTST.")
    if data is None: st.info("Bitte zuerst Daten herunterladen und Experimente ausführen.")
    else:
        c1,c2,c3=st.columns(3); c1.metric("Beginn",str(data.index.min().date())); c2.metric("Ende",str(data.index.max().date())); c3.metric("Letzter Preis",f"{data.gold_usd.iloc[-1]:,.2f} USD")
with tabs[1]:
    st.subheader("Datenexploration")
    if data is not None:
        st.line_chart(data.gold_usd); st.dataframe(pd.DataFrame({"Fehlende Werte":data.isna().sum(),"Abdeckung in %":100*(1-data.isna().mean())})); st.dataframe(data.corr(numeric_only=True).style.background_gradient(cmap="RdBu",vmin=-1,vmax=1))
    else: st.info("Keine aufbereiteten Daten gefunden.")
with tabs[2]:
    files=sorted((PREDICTIONS/"holdout").glob("*.csv")) if (PREDICTIONS/"holdout").exists() else []
    if not files: st.info("Bitte zuerst das Holdout-Experiment ausführen.")
    else:
        selected=st.selectbox("Modell",files,format_func=lambda p:p.stem); prediction=pd.read_csv(selected,parse_dates=["date"],index_col="date"); st.line_chart(prediction); st.download_button("Prognose herunterladen",prediction.to_csv().encode(),selected.name,"text/csv")
        metric_files=list((METRICS/"holdout").glob("*.csv")); metrics=pd.concat([pd.read_csv(p) for p in metric_files],ignore_index=True) if metric_files else pd.DataFrame(); st.dataframe(metrics.sort_values(["horizon","mae"]) if not metrics.empty else metrics)
        if "actual" in prediction: st.line_chart((prediction.actual-prediction.predicted).rename("Residuum"))
with tabs[3]:
    files=sorted((PREDICTIONS/"future").glob("all_models_*.csv")) if (PREDICTIONS/"future").exists() else []
    if not files: st.info("Bitte zuerst die Zukunftsprognose ausführen.")
    else:
        forecast=pd.read_csv(files[-1],parse_dates=["date"],index_col="date"); horizon=st.select_slider("Handelstage",options=[7,30,90],value=30); st.line_chart(forecast.iloc[:horizon]); model=st.selectbox("Einzelmodell",forecast.columns); st.line_chart(forecast[[model]].iloc[:horizon]); st.dataframe(forecast.iloc[:horizon]); st.download_button("CSV herunterladen",forecast.iloc[:horizon].to_csv().encode(),"gold_prognose.csv","text/csv"); st.caption(f"Erstellt aus lokalem Artefakt: {files[-1].name}. Univariate Methode ohne unbekannte zukünftige exogene Variablen.")
with tabs[4]: st.write("Lokale Oberfläche starten: `uv run python scripts/launch_mlflow.py`. Tracking-Daten liegen unter `artifacts/mlflow/`.")
