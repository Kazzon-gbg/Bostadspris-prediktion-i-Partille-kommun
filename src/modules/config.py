from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "partille_housing_real_2023_today.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_PATH = PROJECT_ROOT / "reports" / "report_3years_manual.md"

TARGET = "final_price_sek"
RANDOM_STATE = 42

USE_ASKING_PRICE_AS_FEATURE = True

MIN_VALID_FINAL_PRICE_SEK = 1_000_000
MAX_VALID_FINAL_PRICE_SEK = 30_000_000

EXCLUDE_FEATURE_COLUMNS = [
    # Identifierare och metadata kan få modellen att memorera objekt i stället
    # för att lära sig generella bostadsmönster.
    "address",
    "\ufeffaddress",
    "source_url",
    "detail_url",
    "sold_date",
    "sale_type",
    "municipality",
    # Dessa fält bygger direkt eller indirekt på slutpriset och skulle därför
    # ge dataläckage om de användes som input till modellen.
    "price_per_m2",
    "price_change_sek",
    "price_change_percent",
    "bid_change_percent",
    # Fält som är helt tomma i nuvarande dataset eller främst beskriver
    # datainsamlingsstatus. De tillför inte stabil information till modellen.
    "energy_class",
    "operating_cost_sek_per_year",
    "ownership_type",
    "days_on_market",
    "has_detail_data",
    "price_outlier_flag",
    "area_outlier_flag",
    "plot_outlier_flag",
]
