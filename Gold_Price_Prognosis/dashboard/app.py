"""German artifact-only dashboard; it never trains models."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gold_forecasting.paths import PROCESSED,PREDICTIONS,METRICS,LOSSES,TRADING,FIGURES
from gold_forecasting.interactive_plots import combined_forecast_figure,error_by_lead_time_figure,loss_curves_figure,portfolio_value_figure
from gold_forecasting.display import display_name,display_namespace,strip_signature

st.set_page_config(page_title="Goldpreisprognose",page_icon="📈",layout="wide")
st.title("Goldpreisprognose")
st.warning("Modellbasierte Prognosen sind unsicher und keine Anlageberatung.")
tabs=st.tabs(["Überblick","Datenexploration","Univariable","Multivariant Forecasting","Handelsbot","MLflow"])
datasets=sorted(PROCESSED.glob("gold_dataset_*.parquet"),key=lambda p:p.stat().st_mtime,reverse=True)
data=pd.read_parquet(datasets[0]) if datasets else None

def _experiment_tab(namespace: str, description: str, future_enabled: bool):
    prediction_dir=PREDICTIONS/namespace; metric_dir=METRICS/namespace; loss_dir=LOSSES/namespace
    files=sorted(prediction_dir.glob("*.csv")) if prediction_dir.exists() else []
    st.caption(description)
    if not files:
        st.info(f"Bitte zuerst das Experiment ausführen (Notebook, Kapitel 4: {display_namespace(namespace)}).")
        return
    metric_files=list(metric_dir.glob("*.csv")) if metric_dir.exists() else []
    metrics=pd.concat([pd.read_csv(p) for p in metric_files],ignore_index=True) if metric_files else pd.DataFrame()
    if not metrics.empty: metrics=metrics.assign(model=metrics["model"].map(display_name))
    st.subheader("Modellvergleich (Testzeitraum)")
    st.dataframe(metrics.sort_values(["horizon","mae"]) if not metrics.empty else metrics)

    st.subheader("Alle Modelle im Vergleich (interaktiv -- Legende anklicken zum Ein-/Ausblenden)")
    predictions={strip_signature(f.stem):pd.read_csv(f,parse_dates=["date"]).set_index("date")[["actual","predicted"]] for f in files}
    combined=pd.DataFrame({"actual":next(iter(predictions.values()))["actual"]})
    for name,frame in predictions.items(): combined[name]=frame["predicted"]
    st.plotly_chart(combined_forecast_figure(combined,f"{display_namespace(namespace)}: Modellvergleich"),use_container_width=True)
    if not metrics.empty:
        st.plotly_chart(error_by_lead_time_figure(metrics,f"{display_namespace(namespace)}: MAE nach Lead-Time"),use_container_width=True)

    loss_files=sorted(loss_dir.glob("*.csv")) if loss_dir.exists() else []
    if loss_files:
        st.subheader("Trainings-/Validierungsverlust (nur Modelle mit iterativem Training)")
        loss_histories={strip_signature(f.stem):pd.read_csv(f).to_dict("list") for f in loss_files}
        st.plotly_chart(loss_curves_figure(loss_histories,f"{display_namespace(namespace)}: Verlust"),use_container_width=True)

    st.subheader("Einzelmodell")
    selected=st.selectbox("Modell",files,format_func=lambda p:display_name(strip_signature(p.stem)),key=f"{namespace}_model")
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
            future_model=st.selectbox("Modell (Zukunft)",future_files,format_func=lambda p:display_name(strip_signature(p.stem)),key="future_model")
            future=pd.read_csv(future_model,parse_dates=["date"],index_col="date")
            horizon=st.select_slider("Handelstage",options=[7,30,90],value=30,key="future_horizon")
            st.line_chart(future.iloc[:horizon])
            st.download_button("Zukunftsprognose herunterladen",future.iloc[:horizon].to_csv().encode(),"zukunftsprognose.csv","text/csv",key="future_download")
            st.caption("Univariate Methode ohne unbekannte zukünftige exogene Variablen -- die einzige hier zulässige echte Zukunftsprognose.")

with tabs[0]:
    st.subheader("Projektüberblick")
    st.write("Zielvariable: täglicher Goldpreis in USD je Feinunze. Experiment 1 (univariat, ohne exogene Variablen): Naiv, gleitender Durchschnitt, SARIMA, PatchTST, Chronos Zero-Shot (Original, Bolt, T5-Base, T5-Large, Bolt-Base). Experiment 2 (multivariat, mit exogenen Variablen): Naiv, gleitender Durchschnitt, SARIMAX, XGBoost (Preisniveau), XGBoost (Differenzen), TFT (nativ, kompakt).")
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
    _experiment_tab("multivariate","Experiment 2: Prognose mit exogenen Variablen aus data.yaml (SARIMAX, XGBoost, XGBoost-Differenzen, TFT). Rückblickender Test mit tatsächlich realisierten exogenen Werten (kein unbekannt-zukünftiges Szenario). XGBoost (Differenzen) sagt Renditen statt Preisniveaus vorher -- siehe Notebook Kapitel 4.2 fuer die Begruendung.",future_enabled=False)
with tabs[4]:
    st.subheader("Handelsbot-Backtest")
    st.caption("Vereinfachte Backtest-Simulation je Modell (Notebook Kapitel 5.2): Startkapital 10.000 USD, immer 100% Bargeld oder 100% Gold, keine Gebuehren. Entscheidung einmal pro rollierendem 20-Tage-Schritt anhand der Prognose fuer das Ende dieses Schritts -- Kauf nur bei prognostizierter Rendite > 5 USD, Verkauf nur bei < -5 USD. Der 'cheater'-Bot kennt die zukuenftigen Realpreise und kann taeglich entscheiden -- eine theoretische Obergrenze, kein echtes Modell, keine Anlageberatung.")
    trading_timeseries_path,trading_summary_path=TRADING/"portfolio_timeseries.csv",TRADING/"summary.csv"
    if not trading_timeseries_path.exists() or not trading_summary_path.exists():
        st.info("Bitte zuerst den Handelsbot-Backtest ausfuehren (Notebook, Kapitel 5.2).")
    else:
        trading_timeseries=pd.read_csv(trading_timeseries_path,parse_dates=["date"])
        trading_summary=pd.read_csv(trading_summary_path).sort_values("pnl",ascending=False)
        trading_starting_capital=float((trading_summary["final_value"]-trading_summary["pnl"]).iloc[0])
        trading_wide=trading_timeseries.pivot(index="date",columns="model",values="portfolio_value")
        st.plotly_chart(portfolio_value_figure(trading_wide,trading_starting_capital,"Portfolio-Wert je Bot"),use_container_width=True)
        pnl_png_files=sorted(FIGURES.glob("pnl_summary_*.png"),key=lambda p:p.stat().st_mtime,reverse=True)
        if pnl_png_files: st.image(str(pnl_png_files[0]))
        st.dataframe(trading_summary)

        st.subheader("Einzelner Bot (inkl. Cheater)")
        trading_png_files=sorted(FIGURES.glob("trading_*.png"))
        if not trading_png_files:
            st.info("Noch keine Handelsbot-Diagramme vorhanden.")
        else:
            selected_bot=st.selectbox("Bot",trading_png_files,format_func=lambda p:display_name(strip_signature(p.stem).removeprefix("trading_")),key="trading_bot_select")
            st.image(str(selected_bot))
with tabs[5]: st.write("Lokale Oberfläche starten: `uv run python scripts/launch_mlflow.py`. Tracking-Daten liegen unter `artifacts/mlflow/`.")
