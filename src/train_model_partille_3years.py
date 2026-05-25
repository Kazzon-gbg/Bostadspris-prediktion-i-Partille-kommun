# Bostadspris-prediktion: Partille kommun
"""
Inlämningsuppgift

Mikael Karlsson 2026 för ITHS Pythonprogrammering för AI-utveckling VT 2026

Skriptet läser in huvuddatafilen med cirka tre års verkliga bostadsförsäljningar
från Partille kommun, skapar tydliga figurer, tränar flera regressionsmodeller
och sparar både Markdown-rapport och PDF-rapport.

Förväntad datafil:
    data/partille_housing_real_2023_today.csv

Target/y:
    final_price_sek

Förbättringar som har lagts till under projektets utveckling:

    - Fler modeller: Linear Regression, Decision Tree, Random Forest och Gradient Boosting.
    - Nya features: total_area_m2, has_asking_price, has_extra_area, has_plot_area, quarter.
    - Tydligare feature-urval för att undvika identifierare, tomma metadatafält och dataläckage.
    - Log-transformerad target för modellerna för att hantera stora prisskillnader bättre.
    - Extra felmått: MAPE, alltså genomsnittligt procentuellt fel.
    - Tydligare rapport med jämförelse mellan modeller.
    - Tydligare korrelationsgraf som visar samband med slutpris i stället för en svårläst heatmap.
"""
import os
from datetime import datetime

from modules import config
from modules.config import PROJECT_ROOT


TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M")
THREE_YEAR_REPORT_PATH = PROJECT_ROOT / "reports" / f"report_3years_{TIMESTAMP}.md"

config.REPORT_PATH = THREE_YEAR_REPORT_PATH

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")

from modules.project import HousingPriceProject


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> None:
    project = HousingPriceProject()
    result = project.run(create_pdf=True)
    project.print_completion_summary(result)


if __name__ == "__main__":
    main()
