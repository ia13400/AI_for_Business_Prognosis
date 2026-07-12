# Project Use Case: Prediction for daily Gold Price

 Final assignment for AI for Business Prognosis course. 
 The materials follow the CRISP-DM process model on real-world data.

---

## Notebook Structure for Folder 'Projekt'
📝 TODO
```
AI_for_Business_Prognosis/
├── pyproject.toml
├── uv.lock
├── README.md
├── Projekt/                                          # placeholder for future PatchTST content
    ├── Folien/
    │   └── STL_Folien.pdf                             # lecture slides
    ├── Lern_Notebook/
    │   ├── STL_Nachfrageanalyse.ipynb                 # guided STL walkthrough in CRISP-DM
    │   └── STL_Playground.ipynb                       # interactive LOESS mechanics explorer

```

---

## File Descriptions
📝 TODO

### Other Files

| File | Description |
|---|---|
| `STL/Folien/STL_Folien.pdf` | Lecture slide deck covering STL theory and its CRISP-DM context. |
| `pyproject.toml` | Project metadata and declared dependencies. |
| `uv.lock` | Exact pinned versions of all dependencies for fully reproducible environments. |
| `PatchTST/` | Reserved for future PatchTST transformer-based forecasting content. |

---

## Requirements

- **Python 3.11** or newer
- **uv** package manager

---

## Step-by-Step Setup

### Step 1 — Install uv

uv is a fast Python package and project manager. Install it once on your machine.

**Windows (PowerShell) — via pip:**
```powershell
pip install uv
```

**Windows (PowerShell) — standalone installer (no Python required):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux — via pip:**
```bash
pip install uv
```

**macOS / Linux — standalone installer:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:
```powershell
uv --version
```

---

### Step 2 — Create the project environment

Open a terminal in the project root folder and run:

```powershell
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, then creates a `.venv` folder containing all dependencies at their exact pinned versions. This only needs internet access on the very first run.

---

### Step 3 — Register the Jupyter kernel

Register the project environment as a kernel so VS Code and Jupyter can find it:

```powershell
uv run python -m ipykernel install --user --name ai-for-business-prognosis --display-name "AI for Business Prognosis"
```

This step only needs to be done **once**.

---

## Running the Notebooks

### Option A — VS Code (recommended)

1. Open the project folder in VS Code.
2. Install the **Jupyter** extension if it is not already installed.
3. Open any `.ipynb` file from the file explorer.
4. Click **Select Kernel** in the top-right corner and choose **AI for Business Prognosis**.
5. Run cells with `Shift+Enter` or use **Run All** from the toolbar.

### Option B — Jupyter in the browser

If you prefer the classic Jupyter interface, first add the `notebook` package:

```powershell
uv add notebook
uv sync
```

Then start the server:

```powershell
uv run jupyter notebook
```

A browser tab will open. Navigate to `STL/Lern_Notebook/` or `STL/Uebung_Notebook/`, open any notebook, and select the **AI for Business Prognosis** kernel when prompted.

---


