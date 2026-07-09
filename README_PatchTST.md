# AI for Business Prognosis Details for Implementing PatchTST

An educational notebook environment for learning **PatchTST** as part of an AI for Business Prognosis course. 
The materials are divided into different parts: lecture slides, teaching notebooks, exercise and training examples. 

---

## Repository Structure - Substructure for PatchTST only

```
AI_for_Business_Prognosis/
└── PatchTST/
    ├── Folien/
    │   └── PatchTST_Folien.pdf                             # PatchTST lecture slides
    ├── Lern_Notebook/
    │   ├── PatchTST_Implementierung.ipynb                  # PatchTST implemented with Tensorflow
    │   └── patchtst_teaching_notebook.ipynb                # PatchTST implemented with Pytorch / HuggingFace
    └── Uebung_Notebook/
        └── patchtst_teaching_notebook.ipynb                # exercise (blank)
    └── Training_Examples/
        ├── PatchTST_Beispiel_Training.ipynb                # PatchTST trained on Etth1 Dataset
        ├── PatchTST_Beispiel_Training_Traffic.ipynb        # PatchTST trained on Traffic Dataset
        └── PatchTST_Trainingsoutput_Traffic_100.ipynb      # training output for Traffic Dataset
```

---

## File Descriptions

### Learning Notebooks — `PatchTST/Lern_Notebook/`

| Notebook | Description |
|---|---|
│ `PatchTST_Implementierung.ipynb` | PatchTST implemented with Tensorflow | 
│ `patchtst_teaching_notebook.ipynb` | PatchTST implemented with Pytorch / HuggingFace | 


### Exercise Notebooks — `PatchTST/Uebung_Notebook/`

| Notebook | Description |
|---|---|
| `stl_student_exercise.ipynb` | Three-part exercise: (1) decompose monthly car-sales data, (2) explore STL parameters interactively with sliders, (3) apply STL to Apple (AAPL) stock prices and evaluate when STL is appropriate. |
| `stl_student_exercise_loesung.ipynb` | Complete solution to the main exercise with explanations. |
| `stl_student_exercise_bonus.ipynb` | Bonus exercise extending the AAPL stock analysis. |
| `stl_student_exercise_bonus_loesung.ipynb` | Complete solution to the bonus exercise. |

### Other Files - 'Folien' and 'Trainig_Examples'

| File | Description |
|---|---|
| `PatchTST/Folien/PatchTST_Folien.pdf` | Lecture slide deck covering PatchTST theory.|
| `PatchTST/Training_Examples/PatchTST_Beispiel_Training.ipynb` |PatchTST trained on Etth1 Dataset |
| `PatchTST/Training_Examples/PatchTST_Beispiel_Training_Traffic.ipynb` |PatchTST trained on Traffic Dataset |
| `PatchTST/Training_Examples/PatchTST_Trainingsoutput_Traffic_100.ipynb` |training output for Traffic Dataset |

---

## Requirements

- **Python 3.11** or newer
- **uv** package manager

> [!IMPORTANT]
> For the Tensorflow notebooks please make sure that you use at least Tensorflow 2.22 on your system.
> To select the correct version of tensorflow for your  device refer to the tensorflow documentation.
> Tensorflow versions vary based on  Neural Processing Unit (NPU), Graphics Processing Unit (GPU) and Central Processing Unit (CPU) and the operation system you are using. 

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

A browser tab will open. Navigate to `PatchTST/Lern_Notebook/` or `PatchTST/Uebung_Notebook/` or `PatchTST/Training_Examples/`, open any notebook, and select the **AI for Business Prognosis** kernel when prompted.

---

## :books: Suggested Learning Path

1. :bookmark: Read the slide deck: `PatchTST/Folien/PatchTST_Folien.pdf`
2. :bookmark: Work through `PatchTST/Lern_Notebook/patchtst_teaching_notebook.ipynb to understand how PatchTST is trained on the traffic dataset
3. :bookmark: Work through `PatchTST_Implementierung.ipynb` to learn about the different layers of PatchTST implemented with Tensorflow. The model is trained on synthetic test data.
4. :bookmark: Attempt `PatchTST/Uebung_Notebook/stl_student_exercise.ipynb` before opening any solution notebooks
5. :bookmark: Check your answers against `PatchTST_student_exercise_loesung.ipynb`
6. :bookmark: Try the bonus exercise in `PatchTST_student_exercise_bonus.ipynb`
7. :bookmark: (Optional) Get a look into the Training_Examples/ folder to see how the model performs on our selected example datasets. Please note that all notebooks with 'output' in the name are training output files. They are not executable.

