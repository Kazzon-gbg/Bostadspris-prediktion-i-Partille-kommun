"""
Kör bostadsprismodellen på en separat CSV med senaste två årens försäljningar.

Det här skriptet ändrar inte huvudprojektets ordinarie datafil. Det pekar bara
om sökvägar under denna körning, så filen är enkel att ta bort senare.

Exempel:
    .venv/bin/python src/train_model_partille_2years.py
"""
import os
from datetime import datetime

from modules import config
from modules.config import PROJECT_ROOT


TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M")
TWO_YEAR_DATA_PATH = PROJECT_ROOT / "data" / "partille_housing_real_last_2_years.csv"
TWO_YEAR_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures_2years"
TWO_YEAR_MODEL_DIR = PROJECT_ROOT / "outputs" / "models_2years"
TWO_YEAR_REPORT_PATH = PROJECT_ROOT / "reports" / f"report_2years_{TIMESTAMP}.md"

config.DATA_PATH = TWO_YEAR_DATA_PATH
config.FIGURE_DIR = TWO_YEAR_FIGURE_DIR
config.MODEL_DIR = TWO_YEAR_MODEL_DIR
config.REPORT_PATH = TWO_YEAR_REPORT_PATH

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")

from modules import functions, report, report_pdf
from modules.project import HousingPriceProject


functions.DATA_PATH = TWO_YEAR_DATA_PATH
functions.FIGURE_DIR = TWO_YEAR_FIGURE_DIR
functions.MODEL_DIR = TWO_YEAR_MODEL_DIR
functions.REPORT_PATH = TWO_YEAR_REPORT_PATH
report.PROJECT_ROOT = PROJECT_ROOT
report.DATA_PATH = TWO_YEAR_DATA_PATH
report.REPORT_PATH = TWO_YEAR_REPORT_PATH
report_pdf.FIGURE_DIR = TWO_YEAR_FIGURE_DIR
report_pdf.DATA_PATH = TWO_YEAR_DATA_PATH
report_pdf.REPORT_PATH = TWO_YEAR_REPORT_PATH


def main() -> None:
    project = HousingPriceProject()
    result = project.run(create_pdf=True)
    project.print_completion_summary(result)


if __name__ == "__main__":
    main()
