import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_text_files_are_valid_utf8_without_mojibake():
    forbidden=("Ã"+"ƒ","f"+"Ã"+"¼r","Gesch"+"Ã")
    for path in [*ROOT.rglob("*.py"),*ROOT.rglob("*.md"),*ROOT.rglob("*.yaml")]:
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts): continue
        text=path.read_text(encoding="utf-8"); assert not any(token in text for token in forbidden),path
def test_notebook_markdown_is_utf8_german():
    notebook=json.loads((ROOT/"notebooks"/"gold_price_forecasting.ipynb").read_text(encoding="utf-8")); markdown="\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="markdown"); assert "Geschäftsverständnis" in markdown and "ü" in markdown and "ß" in markdown; assert "Ã" not in markdown
