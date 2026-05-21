# Bostadspris-prediktion: Partille / Sävedalen - förbättrad modellversion
"""
Inlämningsuppgift

Mikael Karlsson 2026 för ITHS Pythonprogrammering för AI-utveckling VT 2026

Skriptet läser in en städad CSV-fil med verkliga bostadsförsäljningar från
Partille kommun, skapar tydliga figurer, tränar flera regressionsmodeller
och sparar en resultatrapport.

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
from modules.config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")

from modules.project import HousingPriceProject


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    project = HousingPriceProject()
    result = project.run(create_pdf=False)
    project.print_completion_summary(result)


if __name__ == "__main__":
    main()
