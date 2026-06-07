# AI for Business Prognosis

This project contains a Jupyter notebook that explains STL decomposition with the AirPassengers time-series dataset.

## Requirements

- Python 3.11 or newer
- `uv`

If `uv` is not installed yet, install it from the official documentation:

```powershell
pip install uv
```

## Setup With uv

From this project folder, install the dependencies into the local virtual environment:

```powershell
uv sync
```

This reads `pyproject.toml` and `uv.lock`, then creates or updates the `.venv` environment.

## Register the Notebook Kernel

Register the project environment as a Jupyter kernel:

```powershell
uv run python -m ipykernel install --user --name ai-for-business-prognosis --display-name "AI for Business Prognosis"
```

After this, Jupyter can use the project's dependencies through the kernel named `AI for Business Prognosis`.

## Run the Notebook

Start Jupyter from the project folder:

```powershell
uv run jupyter notebook
```

Then open:

```text
stl_method_airpassengers_example.ipynb
```

In the notebook, select the kernel `AI for Business Prognosis` and run the cells from top to bottom.

## If Jupyter Is Missing

If the `jupyter notebook` command is not available, add Jupyter to the project:

```powershell
uv add jupyter
uv sync
```

Then start the notebook again:

```powershell
uv run jupyter notebook
```

## Internet Access Note

The notebook first tries to load the AirPassengers dataset through `statsmodels`. If that does not work because your environment has no internet access, use the embedded fallback data cell inside the notebook.
