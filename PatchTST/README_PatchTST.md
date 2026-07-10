# :thought_balloon: AI for Business Prognosis Details for the PatchTST Structure

An educational notebook environment for learning **PatchTST** as part of an AI for Business Prognosis course. 
The materials are divided into different parts: lecture slides, teaching notebooks, exercise and training examples. 

---

## :card_index_dividers: Repository Structure - Substructure for PatchTST only

```
AI_for_Business_Prognosis/
└── PatchTST/
    ├── Folien/
        └── PatchTST_Folien.pdf                             # PatchTST lecture slides
    ├── Lern_Notebook/
        ├── PatchTST_Implementierung.ipynb                  # PatchTST implemented with Tensorflow
        └── patchtst_teaching_notebook.ipynb                # PatchTST implemented with Pytorch / HuggingFace
    └── Uebung_Notebook/
        └── PatchTST_student_exercise.ipynb                 # exercise (blank)
        └── PatchTST_student_exercise_loesung.ipynb         # solution for the exercise
        └── data /
            └── weather.csv                                 # weather data set for the exercise
        └── trained_models_loesung /                        
            └── model_context_336_pred_48.pt                # trained modell example
            └── model_context_336_pred_24.pt                # trained modell example
            └── model_context_168_pred_48.pt                # trained modell example
            └── model_context_168_pred_24.pt                # trained modell example
            └── learning_curves.pt                          # learning curve
        └── trained_models_exercise /                       # A place for your trained models
    └── Training_Examples/
        ├── PatchTST_Beispiel_Training_Traffic.ipynb        # PatchTST trained on Traffic Dataset
        └── PatchTST_Trainingsoutput_Traffic_100.ipynb      # training output for Traffic Dataset
```

---

## :card_index_dividers: File Descriptions

### :clipboard: Learning Notebooks — `PatchTST/Lern_Notebook/`

| Notebook | Description |
|---|---|
│ `PatchTST_Implementierung.ipynb` | PatchTST implemented with Tensorflow | 
│ `patchtst_teaching_notebook.ipynb` | PatchTST implemented with PyTorch / HuggingFace | 


### :clipboard: Exercise Notebooks — `PatchTST/Uebung_Notebook/`

| Notebook | Description |
|---|---|
| `PatchTST_student_exercise.ipynb` | Exercise for learning PatchTST |
| `PatchTST_student_exercise_loesung.ipynb` | Complete solution to the exercise with explanations. |
| `data/weather.csv` | weather data set for the exercise  |
| `trained_models_loesung/model_context_336_pred_48.pt` | trained modell example |
| `trained_models_loesung/model_context_336_pred_24.pt` | trained modell example |
| `trained_models_loesung/model_context_168_pred_48.pt` | trained modell example |
| `trained_models_loesung/model_context_168_pred_24.pt` | trained modell example |
| `trained_models_loesung/learning_curves.pt` | learning curve |
| `trained_models_exercise/` | A folder where your trained models belong |


### :clipboard: Other Files - 'Folien' and 'Training_Examples'

| File | Description |
|---|---|
| `PatchTST/Folien/PatchTST_Folien.pdf` | Lecture slide deck covering PatchTST theory.|
| `PatchTST/Training_Examples/PatchTST_Beispiel_Training_Traffic.ipynb` |PatchTST trained on Traffic Dataset |
| `PatchTST/Training_Examples/PatchTST_Trainingsoutput_Traffic_100.json` |training output for Traffic Dataset |

---

## :gear: Requirements

- **Python 3.11** or newer
- **uv** package manager

> [!IMPORTANT]
> For the Tensorflow notebooks please make sure that you use at least Tensorflow 2.22 on your system.
> To select the correct version of tensorflow for your  device refer to the tensorflow documentation.
> Tensorflow versions vary based on  Neural Processing Unit (NPU), Graphics Processing Unit (GPU) and Central Processing Unit (CPU) and the operating system you are using. 

---

## :hammer_and_wrench: Step-by-Step Setup

### :hammer: Step 1 — Install uv

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

### :hammer: Step 2 — Create the project environment

Open a terminal in the project root folder and run:

```powershell
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, then creates a `.venv` folder containing all dependencies at their exact pinned versions. This only needs internet access on the very first run.

---

### :hammer: Step 3 — Register the Jupyter kernel

Register the project environment as a kernel so VS Code and Jupyter can find it:

```powershell
uv run python -m ipykernel install --user --name ai-for-business-prognosis --display-name "AI for Business Prognosis"
```

This step only needs to be done **once**.

---

## :car: Running the Notebooks

### :wheel: Option A — VS Code (recommended)

1. Open the project folder in VS Code.
2. Install the **Jupyter** extension if it is not already installed.
3. Open any `.ipynb` file from the file explorer.
4. Click **Select Kernel** in the top-right corner and choose **AI for Business Prognosis**.
5. Run cells with `Shift+Enter` or use **Run All** from the toolbar.

### :wheel: Option B — Jupyter in the browser

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
2. :bookmark: Work through `PatchTST/Lern_Notebook/patchtst_teaching_notebook.ipynb` to understand how PatchTST is trained on the traffic dataset
3. :bookmark: Work through `PatchTST_Implementierung.ipynb` to learn about the different layers of PatchTST implemented with Tensorflow. The model is trained on a synthetic data set.
4. :bookmark: Attempt `PatchTST/Uebung_Notebook/PatchTST_student_exercise.ipynb` before opening any solution notebooks
5. :bookmark: Check your answers against `PatchTST_student_exercise_loesung.ipynb`
6. :bookmark: (Optional) Take a look into the Training_Examples/ folder to see how the model performs on a larger typical timeseries dataset. Please note that the json file located in the directory is not executable.

