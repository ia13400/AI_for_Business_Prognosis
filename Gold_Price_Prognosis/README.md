# Reproduzierbare Goldpreisprognose

Dieses Projekt prognostiziert tägliche Goldpreise in USD je Feinunze über zwei Experimente: Experiment 1 (univariat) vergleicht SARIMA, PatchTST und Chronos (Original & Bolt, Zero-Shot) gegen Persistenz und gleitenden Durchschnitt; Experiment 2 (multivariat) vergleicht SARIMAX, XGBoost und ein kompaktes, nativ implementiertes TFT unter Einbeziehung exogener Variablen. Validierung und Test sind je 2.5 Jahre lang, chronologisch vom Enddatum der Daten rückwärts abgeschnitten, und beides ist konfigurierbar.

> Sämtliche Prognosen sind modellbasierte Schätzungen, hochgradig unsicher und keine Anlageberatung.

## Architektur und Methodik

- `src/gold_forecasting/`: Download, Validierung, Aufbereitung, Hashing, Cache, Feature-Engineering, Hyperparameteroptimierung (Optuna), Modelle, Metriken, Diagramme, MLflow und Experiment-Orchestrierung
- `src/gold_forecasting/models/`: eine Datei pro Modell (`sarima.py`, `patchtst.py`, `chronos_zero_shot.py`, `sarimax.py`, `xgboost_model.py`, `tft.py`, `naive.py`, `moving_average.py`); jede Datei enthält ihre komplette Fit-/Predict-/HPO-Logik und stellt eine `run_<modell>(...)`-Funktion bereit
- `configs/`: zentrale Daten-, Modell-, Experiment- und Dashboardparameter (u. a. Split-Länge, Enddatum, HPO-Suchräume)
- `scripts/`: kleine Kommandozeilen-Einstiegspunkte
- `notebooks/gold_price_forecasting.ipynb`: deutsches CRISP-DM-Notebook; ruft ausschließlich Paketfunktionen auf, keine Modell-Logik im Notebook selbst
- `dashboard/`: rein artefaktbasiertes Streamlit-Dashboard mit getrennten Tabs "Univariable" und "Multivariant Forecasting"; kein implizites Training
- `tests/`: Hash-, Cache-, Leakage-, Metrik-, UTF-8-, Modell-, HPO- und End-to-End-Tests

Test bleibt während der gesamten Modellierung strikt verborgen; Validierung dient ausschließlich der Hyperparameteroptimierung und dem Early Stopping. Mehrschrittprognosen werden rekursiv vom eingefrorenen Ursprung (Ende von Training+Validierung) erzeugt, ohne echte versteckte Zielwerte einzuspeisen. Experiment 2 verwendet für den Testzeitraum die tatsächlich realisierten exogenen Werte (Standard-Backtesting-Annahme für Modelle mit exogenen Regressoren, siehe `models/sarimax.py`); echte Zukunftsprognosen (Kapitel 6 im Notebook, `scripts/run_future_forecast.py`) bleiben deshalb auf die univariaten Experiment-1-Modelle beschränkt.

Optuna-Studien werden inhaltsadressiert unter `artifacts/optuna/*.db` gespeichert (SQLite): Ein unterbrochener Lauf führt beim nächsten Aufruf nur die noch fehlenden Trials aus. Neuronale Modelle (PatchTST, TFT) speichern zusätzlich nach jeder Epoche einen Checkpoint und setzen ein unterbrochenes finales Training an der letzten abgeschlossenen Epoche fort.

## Installation mit der vorhandenen uv-Umgebung

Vom übergeordneten Repository:

```powershell
uv sync
```

Python 3.11 oder eine kompatible neuere Version wird unterstützt. Geräteauswahl erfolgt automatisch in der Reihenfolge CUDA, Apple MPS, CPU. Chronos lädt vortrainierte Gewichte bei Bedarf von Hugging Face herunter (Internetzugriff beim ersten Lauf erforderlich, danach lokal zwischengespeichert).

## Daten und Cache

```powershell
uv run python Gold_Price_Prognosis/scripts/download_data.py
uv run python Gold_Price_Prognosis/scripts/download_data.py --force-download
```

Yahoo-Finance-Rohdaten werden einmalig unter `data/raw/` gespeichert. Metadaten enthalten Quelle, Symbol, UTC-Downloadzeit, angefragten Zeitraum, Zeilen, Spalten und SHA-256. Der Goldzielwert wird nicht über fehlende Markttage aufgefüllt. Exogene Reihen werden am Gold-Handelskalender ausgerichtet und nur vergangenheitsbasiert bis maximal drei Zieltermine fortgeschrieben. Das Enddatum ist in `configs/data.yaml: end` konfigurierbar. Verarbeitete Daten liegen als Parquet unter `data/processed/`.

Jeder relevante Lauf erhält eine SHA-256-Signatur aus Dateninhalt, Namensraum, Datumsbereichen, Modell, Hyperparametern, Seed und Horizont. Manifeste liegen neben Artefakten. Geänderte Eingaben erzeugen automatisch neue Artefakte; die Namensräume `univariate`, `multivariate` und `future` sind getrennt. `--force-retrain` umgeht Modell- und Prognosewiederverwendung.

## Experimente

Schneller Entwicklungslauf:

```powershell
uv run python Gold_Price_Prognosis/scripts/run_holdout_experiment.py --mode quick_mode
uv run python Gold_Price_Prognosis/scripts/run_future_forecast.py --horizon 90 --mode quick_mode
```

Finaler Lauf:

```powershell
uv run python Gold_Price_Prognosis/scripts/run_holdout_experiment.py --mode full_mode --force-retrain
uv run python Gold_Price_Prognosis/scripts/run_future_forecast.py --horizon 90 --mode full_mode --force-retrain
```

Der Split (Validierungs-/Testlänge in Jahren) steht in `configs/experiments.yaml: split`. Prognosen, Kennzahlen, Checkpoints und PNGs erscheinen unter `artifacts/`. MAE, RMSE, MASE, sMAPE und Richtungsgenauigkeit werden für 1, 7, 30, 90 Tage und den vollständigen Testzeitraum berechnet. Naive Persistenz und gleitender Durchschnitt sind immer Teil des Vergleichs.

## Notebook, MLflow und Dashboard

```powershell
uv run jupyter lab Gold_Price_Prognosis/notebooks/gold_price_forecasting.ipynb
uv run python Gold_Price_Prognosis/scripts/launch_mlflow.py
uv run python Gold_Price_Prognosis/scripts/launch_dashboard.py
```

MLflow nutzt einen SQLite-Tracking-Store unter `artifacts/mlflow/mlflow.db` mit getrennten Experimenten `gold-univariate` und `gold-multivariate`. Die Oberfläche läuft standardmäßig auf `http://127.0.0.1:5000`. Das Dashboard zeigt zwei Experiment-Tabs ("Univariable", "Multivariant Forecasting") sowie Überblick, Datenexploration und MLflow-Hinweise; fehlende Artefakte werden als verständlicher Hinweis angezeigt.

## Tests

```powershell
uv run pytest
```

Die Tests benötigen für den Standardlauf kein Netzwerk, mit Ausnahme des Chronos-Zero-Shot-Smoke-Tests (lädt `amazon/chronos-bolt-small` einmalig herunter, danach lokal zwischengespeichert). Sie prüfen insbesondere den chronologischen Split, zeitliche Reihenfolge, vergangenheitsbasierte Features (inklusive exogener Variablen), Content-Hashes, Cache-Manifeste, Optuna-Resume-Verhalten, Metriken, alle Modell-Schnittstellen, UTF-8 und deutsche Umlaute.

## Grenzen und Risiken

- Futures-Preise sind nicht identisch mit Spotpreisen und können Roll- und Laufzeiteffekte enthalten.
- Eine einzelne Testperiode deckt nicht alle Marktregime ab.
- Punktprognosen bilden Parameter-, Modell- und Ereignisunsicherheit unvollständig ab.
- Rekursive Mehrschrittprognosen akkumulieren Fehler.
- Experiment 2 setzt für den Testzeitraum tatsächlich realisierte exogene Werte voraus (kein unbekannt-zukünftiges Szenario); eine echte Zukunftsprognose mit exogenen Variablen würde deren eigene Prognose erfordern, was hier bewusst außerhalb des Projektumfangs bleibt.
- Yahoo Finance kann Historien nachträglich korrigieren; der lokale Hash macht den verwendeten Stand nachvollziehbar.
- Gute historische Kennzahlen garantieren weder zukünftige Güte noch Profitabilität. Transaktionskosten, Liquidität, Risiko und regulatorische Anforderungen sind nicht Bestandteil des Modells.
