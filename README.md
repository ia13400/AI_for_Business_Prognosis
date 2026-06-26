# AI for Business Prognosis

An educational notebook environment for learning **STL (Seasonal and Trend decomposition using LOESS)** as part of an AI for Business Prognosis course. The materials follow the CRISP-DM process model and progress from foundational LOESS theory to hands-on exercises on real-world data.

---

## Repository Structure

```
AI_for_Business_Prognosis/
├── pyproject.toml
├── uv.lock
├── README.md
├── PatchTST/                                          # placeholder for future PatchTST content
└── STL/
    ├── Folien/
    │   └── STL_Folien.pdf                             # lecture slides
    ├── Lern_Notebook/
    │   ├── STL_Nachfrageanalyse.ipynb                 # guided STL walkthrough in CRISP-DM
    │   └── STL_Playground.ipynb                       # interactive LOESS mechanics explorer
    └── Uebung_Notebook/
        ├── stl_student_exercise.ipynb                 # student exercise (blank)
        ├── stl_student_exercise_loesung.ipynb         # exercise solution
        ├── stl_student_exercise_bonus.ipynb           # bonus exercise (blank)
        └── stl_student_exercise_bonus_loesung.ipynb   # bonus solution
```

---

## File Descriptions

### Learning Notebooks — `STL/Lern_Notebook/`

| Notebook | Description |
|---|---|
| `STL_Nachfrageanalyse.ipynb` | Step-by-step STL walkthrough embedded in the CRISP-DM framework. Uses synthetic monthly demand data (2018–2023) to demonstrate standard vs. robust STL, residual-based outlier detection, and when to trigger a second modelling iteration. |
| `STL_Playground.ipynb` | Deep dive into LOESS — the core algorithm behind STL. Explores how span size, tricube weighting, and robust weights affect the smoother, before those concepts are applied to time series decomposition. |

### Exercise Notebooks — `STL/Uebung_Notebook/`

| Notebook | Description |
|---|---|
| `stl_student_exercise.ipynb` | Three-part exercise: (1) decompose monthly car-sales data, (2) explore STL parameters interactively with sliders, (3) apply STL to Apple (AAPL) stock prices and evaluate when STL is appropriate. |
| `stl_student_exercise_loesung.ipynb` | Complete solution to the main exercise with explanations. |
| `stl_student_exercise_bonus.ipynb` | Bonus exercise extending the AAPL stock analysis. |
| `stl_student_exercise_bonus_loesung.ipynb` | Complete solution to the bonus exercise. |

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

## Suggested Learning Path

1. Read the slide deck: `STL/Folien/STL_Folien.pdf`
2. Work through `STL/Lern_Notebook/STL_Playground.ipynb` to understand how LOESS works
3. Work through `STL/Lern_Notebook/STL_Nachfrageanalyse.ipynb` to see STL applied in a full CRISP-DM cycle
4. Attempt `STL/Uebung_Notebook/stl_student_exercise.ipynb` before opening any solution notebooks
5. Check your answers against `stl_student_exercise_loesung.ipynb`
6. Try the bonus exercise in `stl_student_exercise_bonus.ipynb`
