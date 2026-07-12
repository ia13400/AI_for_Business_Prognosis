# Reproduzierbare Goldpreisprognose

Dieses Projekt prognostiziert tägliche Goldpreise in USD je Feinunze mit Persistenz, ARIMA und den nativen PyTorch-Modellen LSTM, N-BEATS und PatchTST. Es umfasst einen strikt leckagefreien historischen Holdout ab `2025-06-01` und univariate Zukunftsprognosen über 7, 30 oder 90 Handelstage.

> Sämtliche Prognosen sind modellbasierte Schätzungen, hochgradig unsicher und keine Anlageberatung.

## Architektur und Methodik

- `src/gold_forecasting/`: Download, Validierung, Aufbereitung, Hashing, Cache, Modelle, Metriken, Diagramme, MLflow und Experimente
- `configs/`: zentrale Daten-, Modell-, Experiment- und Dashboardparameter
- `scripts/`: kleine Kommandozeilen-Einstiegspunkte
- `notebooks/gold_price_forecasting.ipynb`: deutsches CRISP-DM-Notebook
- `dashboard/`: rein artefaktbasiertes Streamlit-Dashboard; kein implizites Training
- `tests/`: Hash-, Cache-, Leakage-, Metrik-, UTF-8-, Modell- und End-to-End-Tests

Im Holdout werden alle Zielwerte ab dem Stichtag verborgen. Training, Validierung, Skalierer und Hyperparameter liegen vollständig davor. Mehrschrittprognosen werden rekursiv vom eingefrorenen Ursprung erzeugt, ohne echte versteckte Zielwerte einzuspeisen. Zukunftsprognosen sind univariat. Damit werden keine unbekannten zukünftigen exogenen Größen vorausgesetzt. Exogene Daten dienen der historischen Exploration und optionalen Erweiterungen.

## Installation mit der vorhandenen uv-Umgebung

Vom übergeordneten Repository:

```powershell
uv sync
```

Python 3.11 oder eine kompatible neuere Version wird unterstützt. Geräteauswahl erfolgt automatisch in der Reihenfolge CUDA, Apple MPS, CPU.

## Daten und Cache

```powershell
uv run python Gold_Price_Prognosis/scripts/download_data.py
uv run python Gold_Price_Prognosis/scripts/download_data.py --force-download
```

Yahoo-Finance-Rohdaten werden einmalig unter `data/raw/` gespeichert. Metadaten enthalten Quelle, Symbol, UTC-Downloadzeit, angefragten Zeitraum, Zeilen, Spalten und SHA-256. Der Goldzielwert wird nicht über fehlende Markttage aufgefüllt. Exogene Reihen werden am Gold-Handelskalender ausgerichtet und nur vergangenheitsbasiert bis maximal drei Zieltermine fortgeschrieben. Verarbeitete Daten liegen als Parquet unter `data/processed/`.

Jeder relevante Lauf erhält eine SHA-256-Signatur aus Dateninhalt, Namensraum, Cutoff, Datumsbereichen, Pipelineversion, Modell, Hyperparametern, Seed und Horizont. Manifeste liegen neben Artefakten. Geänderte Eingaben erzeugen automatisch neue Artefakte; Holdout und Zukunft nutzen getrennte Namensräume. `--force-retrain` umgeht Modell- und Prognosewiederverwendung.

## Experimente

Schneller Entwicklungslauf:

```powershell
uv run python Gold_Price_Prognosis/scripts/run_holdout_experiment.py --mode quick_mode
uv run python Gold_Price_Prognosis/scripts/run_future_forecast.py --horizon 90 --mode quick_mode
```

Finaler Lauf:

```powershell
uv run python Gold_Price_Prognosis/scripts/run_holdout_experiment.py --cutoff 2025-06-01 --mode full_mode --force-retrain
uv run python Gold_Price_Prognosis/scripts/run_future_forecast.py --horizon 90 --mode full_mode --force-retrain
```

Prognosen, Kennzahlen, Checkpoints und PNGs erscheinen unter `artifacts/`. MAE, RMSE, MASE, sMAPE und Richtungsgenauigkeit werden für 1, 7, 30 Tage und den vollständigen Holdout berechnet. Die naive Persistenz ist immer Teil des Vergleichs. Neuronale Modelle verwenden Early Stopping und speichern den besten Validierungszustand.

## Notebook, MLflow und Dashboard

```powershell
uv run jupyter lab Gold_Price_Prognosis/notebooks/gold_price_forecasting.ipynb
uv run python Gold_Price_Prognosis/scripts/launch_mlflow.py
uv run python Gold_Price_Prognosis/scripts/launch_dashboard.py
```

MLflow verwendet standardmäßig `artifacts/mlflow/` und getrennte Experimente `gold-holdout` und `gold-future`. Die Oberfläche läuft standardmäßig auf `http://127.0.0.1:5000`. Das Dashboard zeigt fehlende Artefakte als verständlichen Hinweis und gibt keine Stacktraces an normale Benutzer aus.

## Tests

```powershell
uv run pytest
```

Die Tests benötigen für den Standardlauf kein Netzwerk. Sie prüfen insbesondere den Stichtag, zeitliche Reihenfolge, vergangenheitsbasierte Features, Content-Hashes, Cache-Manifeste, Metriken, alle Modell-Schnittstellen, UTF-8 und deutsche Umlaute.

## Grenzen und Risiken

- Futures-Preise sind nicht identisch mit Spotpreisen und können Roll- und Laufzeiteffekte enthalten.
- Ein einzelner Holdout deckt nicht alle Marktregime ab.
- Punktprognosen bilden Parameter-, Modell- und Ereignisunsicherheit unvollständig ab.
- Rekursive Mehrschrittprognosen akkumulieren Fehler.
- Yahoo Finance kann Historien nachträglich korrigieren; der lokale Hash macht den verwendeten Stand nachvollziehbar.
- Gute historische Kennzahlen garantieren weder zukünftige Güte noch Profitabilität. Transaktionskosten, Liquidität, Risiko und regulatorische Anforderungen sind nicht Bestandteil des Modells.
