"""
Kör hela modellflödet och skapa en tidsstämplad PDF-rapport i reports/.

Exempel:
    .venv/bin/python src/train_model_partille_toPDF.py
"""
import os

from modules.config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")

from modules.project import HousingPriceProject


def main() -> None:
    project = HousingPriceProject()
    result = project.run(create_pdf=True)
    project.print_completion_summary(result)


if __name__ == "__main__":
    main()
