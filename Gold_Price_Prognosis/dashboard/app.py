"""German artifact-only dashboard; it never trains models."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gold_forecasting.paths import PROCESSED,PREDICTIONS,METRICS,LOSSES,TRADING,FIGURES
from gold_forecasting.interactive_plots import combined_forecast_figure,error_by_lead_time_figure,loss_curves_figure,portfolio_value_figure,exogenous_overview_figure
from gold_forecasting.display import display_name,display_namespace,strip_signature

st.set_page_config(page_title="Goldpreisprognose",page_icon="📈",layout="wide")
st.title("Goldpreisprognose")
st.warning("Modellbasierte Prognosen sind unsicher und keine Anlageberatung.")
tabs=st.tabs(["Überblick","Datenexploration","Univariable","Multivariant Forecasting","Handelsbot"])
datasets=sorted(PROCESSED.glob("gold_dataset_*.parquet"),key=lambda p:p.stat().st_mtime,reverse=True)
data=pd.read_parquet(datasets[0]) if datasets else None

def _experiment_tab(namespace: str, description: str):
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
    st.caption("Fehlermetriken je Modell und Lead-Time-Stützstelle (Tag 1, 10, 20) sowie eine 'complete'-Zeile über den gesamten rollierenden Testzeitraum. Niedrigere Werte sind besser, außer bei der Richtungsgenauigkeit (höher ist besser).")
    st.dataframe(metrics.sort_values(["horizon","mae"]) if not metrics.empty else metrics)

    st.subheader("Alle Modelle im Vergleich (interaktiv -- Legende anklicken zum Ein-/Ausblenden)")
    st.caption("Tatsächlicher Goldpreis und die Prognose jedes Modells über den gesamten Testzeitraum, sowie der mittlere absolute Fehler (MAE) nach Lead-Time-Tag innerhalb des rollierenden 20-Tage-Fensters.")
    predictions={strip_signature(f.stem):pd.read_csv(f,parse_dates=["date"]).set_index("date")[["actual","predicted"]] for f in files}
    combined=pd.DataFrame({"actual":next(iter(predictions.values()))["actual"]})
    for name,frame in predictions.items(): combined[name]=frame["predicted"]
    st.plotly_chart(combined_forecast_figure(combined,f"{display_namespace(namespace)}: Modellvergleich"),use_container_width=True)
    if not metrics.empty:
        st.plotly_chart(error_by_lead_time_figure(metrics,f"{display_namespace(namespace)}: MAE nach Lead-Time"),use_container_width=True)

    loss_files=sorted(loss_dir.glob("*.csv")) if loss_dir.exists() else []
    if loss_files:
        st.subheader("Trainings-/Validierungsverlust (nur Modelle mit iterativem Training)")
        st.caption("Verlustkurve über alle rollierenden Fenster hinweg aneinandergereiht (Erstanpassung + jedes Warm-Start-Update). Eine wachsende Lücke zwischen Trainings- und Validierungsverlust deutet auf Overfitting hin.")
        loss_histories={strip_signature(f.stem):pd.read_csv(f).to_dict("list") for f in loss_files}
        st.plotly_chart(loss_curves_figure(loss_histories,f"{display_namespace(namespace)}: Verlust"),use_container_width=True)

    st.subheader("Einzelmodell")
    st.caption("Prognose und Residuum (Tatsächlich - Prognose) eines einzelnen Modells, zum Herunterladen als CSV.")
    selected=st.selectbox("Modell",files,format_func=lambda p:display_name(strip_signature(p.stem)),key=f"{namespace}_model")
    prediction=pd.read_csv(selected,parse_dates=["date"]).set_index("date")[["actual","predicted"]]
    st.line_chart(prediction)
    st.download_button("Prognose herunterladen",prediction.to_csv().encode(),selected.name,"text/csv",key=f"{namespace}_download")
    if "actual" in prediction.columns and "predicted" in prediction.columns:
        st.line_chart((prediction["actual"]-prediction["predicted"]).rename("Residuum"))
    st.subheader(f"Echte Zukunftsprognose ({display_namespace(namespace)}-Modelle)")
    model_names={strip_signature(f.stem) for f in files}
    future_files=sorted((PREDICTIONS/"future").glob("*.csv")) if (PREDICTIONS/"future").exists() else []
    future_files=[f for f in future_files if strip_signature(f.stem) in model_names]
    if not future_files:
        st.info("Bitte zuerst die Zukunftsprognose ausführen (Notebook, Kapitel 6).")
    else:
        future_model=st.selectbox("Modell (Zukunft)",future_files,format_func=lambda p:display_name(strip_signature(p.stem)),key=f"{namespace}_future_model")
        future=pd.read_csv(future_model,parse_dates=["date"],index_col="date")
        horizon=st.select_slider("Handelstage",options=[7,30,90],value=30,key=f"{namespace}_future_horizon")
        st.line_chart(future.iloc[:horizon])
        st.download_button("Zukunftsprognose herunterladen",future.iloc[:horizon].to_csv().encode(),"zukunftsprognose.csv","text/csv",key=f"{namespace}_future_download")
        if namespace=="univariate":
            st.caption("Univariate Methode ohne exogene Variablen -- benötigt keine Annahme über zukünftige exogene Werte.")
        else:
            st.caption("Multivariate Methode -- exogene Variablen werden für den gesamten Prognosehorizont auf ihrem letzten bekannten Wert eingefroren (Persistenz-Annahme, keine echte Prognose der exogenen Variablen). Zusätzliche Unsicherheitsquelle gegenüber der univariaten Zukunftsprognose.")

with tabs[0]:
    st.subheader("Projektüberblick")
    st.write("Zielvariable: täglicher Goldpreis in USD je Feinunze. Experiment 1 (univariat, ohne exogene Variablen): Naiv, gleitender Durchschnitt, SARIMA, PatchTST, Chronos (T5, Large) Zero-Shot. Experiment 2 (multivariat, mit exogenen Variablen): Naiv, gleitender Durchschnitt, SARIMAX, XGBoost (Preisniveau), XGBoost (Differenzen), TFT (nativ, kompakt), Chronos-2 Zero-Shot (kovariat-bewusst). Die übrigen vier univariaten Chronos-Varianten sind deaktiviert (konfigurierbar in `configs/models.yaml`) -- je Chronos-Generation bleibt genau ein Modell aktiv.")
    st.write("Validierungs- und Testzeitraum sind je 1 Jahr lang; die Auswertung erfolgt rollierend (walk-forward, Standard: 20 Tage Horizont, 20 Tage Schritt). Alle Werte sind in `configs/experiments.yaml` (`split`, `rolling`) konfigurierbar; das Enddatum der heruntergeladenen Daten steht in `configs/data.yaml`.")
    if data is None: st.info("Bitte zuerst Daten herunterladen und Experimente ausführen.")
    else:
        st.caption("Zeitliche Abdeckung des aufbereiteten Datensatzes und der zuletzt beobachtete Goldpreis.")
        c1,c2,c3=st.columns(3); c1.metric("Beginn",str(data.index.min().date())); c2.metric("Ende",str(data.index.max().date())); c3.metric("Letzter Preis",f"{data.gold_usd.iloc[-1]:,.2f} USD")
        exogenous_columns=[c for c in data.columns if c != "gold_usd"]
        if exogenous_columns:
            st.subheader("Exogene Variablen im Zeitverlauf")
            st.caption("Rohhistorie jeder exogenen Variable aus `configs/data.yaml: exogenous` (US-Dollar-Index, Silber, Öl, S&P 500, VIX, Bitcoin) -- die Grundlage für Experiment 2 (multivariat).")
            st.plotly_chart(exogenous_overview_figure(data,exogenous_columns,"Exogene Variablen im Zeitverlauf"),use_container_width=True)
with tabs[1]:
    st.subheader("Datenexploration")
    if data is not None:
        st.caption("Täglicher Goldpreis (Zielvariable) über die gesamte verfügbare Historie.")
        st.line_chart(data.gold_usd)
        st.caption("Fehlende Werte je Spalte -- exogene Variablen werden nur begrenzt vorwärts aufgefüllt (siehe `data_preparation.py`), der Goldpreis selbst nie.")
        st.dataframe(pd.DataFrame({"Fehlende Werte":data.isna().sum(),"Abdeckung in %":100*(1-data.isna().mean())}))
        st.caption("Pearson-Korrelation zwischen allen Spalten über die gesamte Historie -- rein deskriptiv, fließt in kein Modelltraining ein.")
        st.dataframe(data.corr(numeric_only=True).style.background_gradient(cmap="RdBu",vmin=-1,vmax=1))
    else: st.info("Keine aufbereiteten Daten gefunden.")
with tabs[2]:
    _experiment_tab("univariate","Experiment 1: Prognose ohne exogene Variablen (SARIMA, PatchTST, Chronos T5 Large) gegen Naiv und gleitenden Durchschnitt.")
with tabs[3]:
    _experiment_tab("multivariate","Experiment 2: Prognose mit exogenen Variablen aus data.yaml (SARIMAX, XGBoost, XGBoost-Differenzen, TFT, Chronos-2 Zero-Shot). Rückblickender Test mit tatsächlich realisierten exogenen Werten (kein unbekannt-zukünftiges Szenario). XGBoost (Differenzen) sagt Renditen statt Preisniveaus vorher -- siehe Notebook Kapitel 4.2 fuer die Begruendung. Chronos-2 ist das einzige Chronos-Modell mit nativer Unterstützung für exogene Variablen und läuft hier zero-shot ohne eigenes Training.")
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
        st.subheader("Portfolio-Wert über die Zeit")
        st.caption("Portfoliowert jedes Bots über den Testzeitraum, gestrichelte Linie = Startkapital zum Vergleich.")
        st.plotly_chart(portfolio_value_figure(trading_wide,trading_starting_capital,"Portfolio-Wert je Bot"),use_container_width=True)
        st.subheader("Endwert je Bot")
        st.caption("Endwert des Portfolios je Bot, Balken beginnen beim Startkapital -- rot = Verlust, blau = Gewinn.")
        pnl_png_files=sorted(FIGURES.glob("pnl_summary_*.png"),key=lambda p:p.stat().st_mtime,reverse=True)
        if pnl_png_files: st.image(str(pnl_png_files[0]))
        st.dataframe(trading_summary)

        st.subheader("Einzelner Bot (inkl. Cheater)")
        st.caption("Goldpreis, Prognose (falls vorhanden), Portfoliowert und die resultierende Cash/Gold-Position eines einzelnen Bots über die Zeit.")
        trading_png_files=sorted(FIGURES.glob("trading_*.png"))
        if not trading_png_files:
            st.info("Noch keine Handelsbot-Diagramme vorhanden.")
        else:
            selected_bot=st.selectbox("Bot",trading_png_files,format_func=lambda p:display_name(strip_signature(p.stem).removeprefix("trading_")),key="trading_bot_select")
            st.image(str(selected_bot))
