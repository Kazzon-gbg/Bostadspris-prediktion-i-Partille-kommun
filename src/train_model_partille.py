# Bostadspris-prediktion: Partille / Sävedalen - förbättrad modellversion
"""
Inlämningsuppgift

<<<<<<< HEAD
Mikael Karlsson 2026 för ITHS Pythonprogrammering för AI utveckling VT 2026
=======
Mikael Karlsson 2026 för ITHS Python för AI
>>>>>>> bdd7caef614f31361000c778bd6b8abf3873c059

Scriptet läser in en städad CSV med verkliga bostadsförsäljningar från
Partille kommun, skapar tydliga figurer, tränar flera regressionsmodeller
och sparar en resultatrapport.

Förväntad datafil:
    data/partille_housing_real_2023_today.csv

Target/y:
    final_price_sek

Förbättringar jämfört med tidigare version:
    - Fler modeller: Linear Regression, Decision Tree, Random Forest och Gradient Boosting.
    - Nya features: total_area_m2, has_asking_price, has_extra_area, has_plot_area, quarter.
    - Tydligare feature-urval för att undvika identifierare, tomma metadatafält och dataläckage.
    - Log-transformerad target för modellerna för att hantera stora prisskillnader bättre.
    - Extra felmått: MAPE, alltså genomsnittligt procentuellt fel.
    - Tydligare rapport med jämförelse mellan modeller.
    - Tydligare korrelationsgraf som visar samband med slutpris i stället för en svårläst heatmap.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# =====================================================================
# KONFIGURATION
# =====================================================================

DATA_PATH = PROJECT_ROOT / "data" / "partille_housing_real_2023_today.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_PATH = PROJECT_ROOT / "reports" / "resultat.md"

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


# =====================================================================
# HJÄLPFUNKTIONER
# =====================================================================

def format_sek(value: float) -> str:
    """Formatera kronor så rapporten blir lättläst."""
    return f"{value:,.0f} kr".replace(",", " ")


def format_msek(value: float) -> str:
    """Formatera kronor som miljoner SEK."""
    return f"{value / 1_000_000:.2f} MSEK"


def ensure_output_dirs() -> None:
    """Skapa mapparna där scriptet sparar rapport, figurer och modellfil."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def add_price_columns_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    """Lägg till prisfält i miljoner SEK för mer läsbara axlar i figurerna."""
    plot_df = df.copy()

    if TARGET in plot_df.columns:
        plot_df["final_price_msek"] = plot_df[TARGET] / 1_000_000

    if "asking_price_sek" in plot_df.columns:
        plot_df["asking_price_msek"] = plot_df["asking_price_sek"] / 1_000_000

    if "price_change_sek" in plot_df.columns:
        plot_df["price_change_msek"] = plot_df["price_change_sek"] / 1_000_000

    return plot_df


def load_data() -> pd.DataFrame:
    """Läs in CSV-filen och gör grundläggande typkonvertering och prisrensning."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Hittar inte datafilen: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    if TARGET not in df.columns:
        raise ValueError(f"Datasetet måste innehålla target-kolumnen '{TARGET}'.")

    numeric_columns = [
        TARGET,
        "asking_price_sek",
        "price_change_sek",
        "price_change_percent",
        "living_area_m2",
        "extra_area_m2",
        "rooms",
        "plot_area_m2",
        "price_per_m2",
        "bid_change_percent",
        "year",
        "month",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TARGET])

    if "sold_date" in df.columns:
        df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")

    rows_before = len(df)
    df = df[
        (df[TARGET] >= MIN_VALID_FINAL_PRICE_SEK)
        & (df[TARGET] <= MAX_VALID_FINAL_PRICE_SEK)
    ].copy()
    rows_after = len(df)

    if rows_before != rows_after:
        print(f"Rensade bort {rows_before - rows_after} rader med orimligt slutpris.")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skapa extra features som kan hjälpa modellen.

    Viktigt:
    Dessa features får inte bygga på slutpriset, eftersom det skulle ge dataläckage.
    """
    df = df.copy()

    if "extra_area_m2" in df.columns:
        df["extra_area_m2_filled_zero"] = df["extra_area_m2"].fillna(0)
    else:
        df["extra_area_m2_filled_zero"] = 0

    if "living_area_m2" in df.columns:
        df["total_area_m2"] = df["living_area_m2"].fillna(0) + df["extra_area_m2_filled_zero"]
    else:
        df["total_area_m2"] = df["extra_area_m2_filled_zero"]

    if "asking_price_sek" in df.columns:
        df["has_asking_price"] = df["asking_price_sek"].notna().astype(int)
    else:
        df["has_asking_price"] = 0

    if "extra_area_m2" in df.columns:
        df["has_extra_area"] = df["extra_area_m2"].notna().astype(int)
    else:
        df["has_extra_area"] = 0

    if "plot_area_m2" in df.columns:
        df["has_plot_area"] = df["plot_area_m2"].notna().astype(int)
    else:
        df["has_plot_area"] = 0

    if "month" in df.columns:
        df["quarter"] = ((df["month"] - 1) // 3 + 1).astype("Int64")
    else:
        df["quarter"] = pd.NA

    # Enkel interaktion: boarea gånger antal rum.
    # Detta kan hjälpa modellen förstå skillnader mellan små/stora bostäder.
    if "living_area_m2" in df.columns and "rooms" in df.columns:
        df["area_rooms_interaction"] = df["living_area_m2"] * df["rooms"]

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Välj modellens X-kolumner och filtrera bort target, läckage och metadata."""
    excluded = set(EXCLUDE_FEATURE_COLUMNS)

    if not USE_ASKING_PRICE_AS_FEATURE:
        excluded.add("asking_price_sek")

    return [
        column
        for column in df.columns
        if column != TARGET and column not in excluded
    ]


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Bygg preprocessing för numeriska och kategoriska features."""
    feature_columns = get_feature_columns(df)
    X = df[feature_columns]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def with_log_target(pipeline: Pipeline) -> TransformedTargetRegressor:
    """
    Log-transformera slutpriset under träning.

    Det gör att modellen delvis optimerar relativa fel i stället för att
    stora dyra bostäder helt dominerar träningen.
    """
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def create_models(preprocessor: ColumnTransformer) -> dict[str, object]:
    """
    Skapa modeller.

    Alla modeller körs med log-transformerad target.
    """
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(
                max_depth=7,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    gradient_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    return {
        "Linear Regression log-target": with_log_target(linear_pipeline),
        "Decision Tree log-target": with_log_target(tree_pipeline),
        "Random Forest log-target": with_log_target(forest_pipeline),
        "Gradient Boosting log-target": with_log_target(gradient_pipeline),
    }


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Beräkna de utvärderingsmått som används i rapporten."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mse**0.5,
        "R2": r2_score(y_test, predictions),
        "MAPE": mean_absolute_percentage_error(y_test, predictions) * 100,
    }


# =====================================================================
# FIGURER
# =====================================================================

def save_basic_visualizations(df: pd.DataFrame) -> None:
    """Skapa de övergripande figurerna som beskriver datasetet."""
    sns.set_theme(style="whitegrid")
    plot_df = add_price_columns_for_plots(df)

    plt.figure(figsize=(9, 5))
    sns.histplot(plot_df["final_price_msek"], kde=True)
    plt.title("Fördelning av slutpriser")
    plt.xlabel("Slutpris, miljoner SEK")
    plt.ylabel("Antal bostäder")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_price_distribution.png", dpi=160)
    plt.close()

    if "living_area_m2" in plot_df.columns:
        plt.figure(figsize=(9, 5))
        hue_column = "property_type" if "property_type" in plot_df.columns else None
        sns.scatterplot(
            data=plot_df,
            x="living_area_m2",
            y="final_price_msek",
            hue=hue_column,
        )
        plt.title("Slutpris jämfört med boarea")
        plt.xlabel("Boarea m²")
        plt.ylabel("Slutpris, miljoner SEK")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "02_living_area_vs_price.png", dpi=160)
        plt.close()
        
    # 3. Tydligare korrelationsgraf:
    # I stället för en stor heatmap med alla numeriska kolumner visar vi
    # endast korrelationen mellan varje numerisk variabel och slutpriset.
    # Det blir mer begripligt för mottagaren av rapporten.
    numeric_df = df.select_dtypes(include="number")
    if TARGET in numeric_df.columns and len(numeric_df.columns) > 1:
        correlation_with_price = (
            numeric_df.corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .dropna()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 8))
        sns.barplot(
            x=correlation_with_price.values,
            y=correlation_with_price.index,
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Korrelation med slutpris")
        plt.xlabel("Korrelation med slutpris")
        plt.ylabel("Variabel")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "03_correlation_with_price.png", dpi=160)
        plt.close()

    if "asking_price_msek" in plot_df.columns:
        df_asking = plot_df.dropna(subset=["asking_price_msek", "final_price_msek"])
        if len(df_asking) > 0:
            plt.figure(figsize=(9, 5))
            hue_column = "property_type" if "property_type" in df_asking.columns else None
            sns.scatterplot(
                data=df_asking,
                x="asking_price_msek",
                y="final_price_msek",
                hue=hue_column,
            )
            min_value = min(df_asking["asking_price_msek"].min(), df_asking["final_price_msek"].min())
            max_value = max(df_asking["asking_price_msek"].max(), df_asking["final_price_msek"].max())
            plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
            plt.title("Utgångspris jämfört med slutpris")
            plt.xlabel("Utgångspris, miljoner SEK")
            plt.ylabel("Slutpris, miljoner SEK")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "06_asking_price_vs_final_price.png", dpi=160)
            plt.close()

    if "property_type" in plot_df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=plot_df, x="property_type", y="final_price_msek")
        plt.title("Slutpris per bostadstyp")
        plt.xlabel("Bostadstyp")
        plt.ylabel("Slutpris, miljoner SEK")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "07_price_by_property_type.png", dpi=160)
        plt.close()

    if "price_change_msek" in plot_df.columns:
        df_change = plot_df.dropna(subset=["price_change_msek"])
        if len(df_change) > 0:
            plt.figure(figsize=(9, 5))
            sns.histplot(df_change["price_change_msek"], kde=True)
            plt.axvline(0, color="red", linestyle="--")
            plt.title("Prisförändring från utgångspris till slutpris")
            plt.xlabel("Prisförändring, miljoner SEK")
            plt.ylabel("Antal bostäder")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "08_price_change_distribution.png", dpi=160)
            plt.close()


def save_prediction_plots(best_model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Spara figurer som visar modellens prediktioner och residualer."""
    predictions = best_model.predict(X_test)
    residuals = y_test - predictions

    y_test_msek = y_test / 1_000_000
    predictions_msek = predictions / 1_000_000
    residuals_msek = residuals / 1_000_000

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test_msek, y=predictions_msek)
    min_value = min(y_test_msek.min(), predictions_msek.min())
    max_value = max(y_test_msek.max(), predictions_msek.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    plt.title("Faktiskt slutpris jämfört med predikterat slutpris")
    plt.xlabel("Faktiskt slutpris, miljoner SEK")
    plt.ylabel("Predikterat slutpris, miljoner SEK")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(residuals_msek, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Residualer: faktiskt slutpris minus predikterat slutpris")
    plt.xlabel("Residual, miljoner SEK")
    plt.ylabel("Antal bostäder")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_residuals.png", dpi=160)
    plt.close()


# =====================================================================
# RAPPORT
# =====================================================================

def get_dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    """Samla nyckeltal som används i den genererade Markdown-rapporten."""
    summary: dict[str, object] = {}
    summary["rows"] = len(df)
    summary["columns"] = len(df.columns)
    summary["median_price"] = df[TARGET].median()
    summary["mean_price"] = df[TARGET].mean()
    summary["min_price"] = df[TARGET].min()
    summary["max_price"] = df[TARGET].max()

    if "sold_date" in df.columns and df["sold_date"].notna().any():
        summary["min_date"] = df["sold_date"].min().date()
        summary["max_date"] = df["sold_date"].max().date()
    else:
        summary["min_date"] = "okänt"
        summary["max_date"] = "okänt"

    if "asking_price_sek" in df.columns:
        summary["asking_price_count"] = int(df["asking_price_sek"].notna().sum())
    else:
        summary["asking_price_count"] = 0

    if "property_type" in df.columns:
        summary["property_types"] = ", ".join(sorted(df["property_type"].dropna().unique()))
        summary["property_type_counts"] = df["property_type"].value_counts().to_string()
    else:
        summary["property_types"] = "saknas"
        summary["property_type_counts"] = "saknas"

    if "is_savedalen" in df.columns:
        summary["savedalen_counts"] = df["is_savedalen"].value_counts().to_string()
    else:
        summary["savedalen_counts"] = "saknas"

    return summary


def build_results_table_for_report(results: pd.DataFrame) -> str:
    """Bygg en Markdown-tabell med modellernas utvärderingsmått."""
    lines = [
        "| Modell | MAE, kr | RMSE, kr | R² | MAPE |",
        "|---|---:|---:|---:|---:|",
    ]

    for model_name, row in results.iterrows():
        mae = f"{row['MAE']:,.0f}".replace(",", " ")
        rmse = f"{row['RMSE']:,.0f}".replace(",", " ")
        r2 = f"{row['R2']:.3f}"
        mape = f"{row['MAPE']:.1f} %"
        lines.append(f"| {model_name} | {mae} | {rmse} | {r2} | {mape} |")

    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    results: pd.DataFrame,
    best_model_name: str,
) -> None:
    """Skriv den slutliga resultatrapporten till reports/resultat.md."""
    missing_values = df.isna().sum()
    best_row = results.loc[best_model_name]
    summary = get_dataset_summary(df)

    asking_price_text = (
        "används som feature"
        if USE_ASKING_PRICE_AS_FEATURE
        else "används inte som feature"
    )

    results_table = build_results_table_for_report(results)

    report = f"""# Resultatrapport: Bostadspris-prediktion i Partille kommun

## 1. Sammanfattning

I detta projekt har jag byggt och jämfört flera maskininlärningsmodeller för att
förutsäga slutpris på bostäder i Partille kommun. Datasetet består av verkliga
bostadsförsäljningar från perioden **{summary["min_date"]} till {summary["max_date"]}**.

<<<<<<< HEAD
Flera regressionsmodeller jämförs för att se vilken metod som bäst fångar
sambanden i bostadsdatan. Linear Regression används som en enkel och tydlig
basmodell, medan Decision Tree, Random Forest och Gradient Boosting kan fånga
mer komplexa och icke-linjära samband mellan bostädernas egenskaper och
slutpris. Slutpriset log-transformeras under träningen för att modellen inte
ska påverkas lika mycket av de dyraste bostäderna.
=======
I den första versionen testades enklare modeller. I denna förbättrade version
används även ensemblemodeller, exempelvis Random Forest och Gradient Boosting.
Dessutom används log-transformering av slutpriset för att modellen inte ska
påverkas lika mycket av de dyraste bostäderna.
>>>>>>> bdd7caef614f31361000c778bd6b8abf3873c059

Den bästa modellen i denna körning blev **{best_model_name}**.

- **MAE:** {format_sek(best_row["MAE"])} ({format_msek(best_row["MAE"])})
- **RMSE:** {format_sek(best_row["RMSE"])} ({format_msek(best_row["RMSE"])})
- **R²:** {best_row["R2"]:.3f}
- **MAPE:** {best_row["MAPE"]:.1f} %

MAPE visar modellens genomsnittliga procentuella fel. Detta är ofta lättare att
tolka än kronor eftersom bostäderna ligger på olika prisnivåer.

## 2. Varför blir modellen inte perfekt?

Även med data från mer än tre år är bostadspriser svåra att prediktera exakt.
Det beror på att viktiga faktorer saknas i datasetet, exempelvis:

- exakt mikroläge
- skick och renoveringsstandard
- planlösning
- utsikt
- energiklass
- närhet till skola, buss, service och natur
- budgivningsläge
- ränteläge och marknadsläge vid försäljningen

Därför är det rimligt att modellen inte hamnar exakt rätt på varje bostad.
Målet är inte att få ett perfekt facit, utan att undersöka om modellen kan hitta
mönster och göra en rimlig uppskattning.

## 3. Syfte

Syftet är att träna och utvärdera regressionsmodeller som predikterar bostadspris.

- **Target/y-värde:** `{TARGET}`
- **Det modellen ska prediktera:** slutpris i kronor
- **Exempel på X-värden/features:** boarea, rum, tomtarea, bostadstyp, område, år, månad och utgångspris

## 4. Dataset

Datasetet består av städad bostadsdata för Partille kommun, inklusive Sävedalen.
Datan omfattar bostadstyperna **{summary["property_types"]}**.

- Datafil: `{DATA_PATH.relative_to(PROJECT_ROOT)}`
- Datumintervall: **{summary["min_date"]} till {summary["max_date"]}**
- Antal rader efter grundstädning: **{summary["rows"]}**
- Antal kolumner efter feature engineering: **{summary["columns"]}**
- Rader med beräknat eller tillgängligt utgångspris: **{summary["asking_price_count"]}**
- Utgångspris i modellen: **{asking_price_text}**

### Fördelning per bostadstyp

```text
{summary["property_type_counts"]}
```

### Sävedalen jämfört med övriga Partille

```text
{summary["savedalen_counts"]}
```

### Prisnivåer

- Lägsta slutpris efter städning: **{format_sek(summary["min_price"])}**
- Högsta slutpris efter städning: **{format_sek(summary["max_price"])}**
- Medianpris: **{format_sek(summary["median_price"])}**
- Genomsnittspris: **{format_sek(summary["mean_price"])}**

## 5. Saknade värden

Verklig bostadsdata innehåller saknade värden. Exempelvis saknas ibland
utgångspris, rum, biarea eller tomtarea.

Saknade numeriska värden ersätts med medianvärde. Saknade kategoriska värden
ersätts med vanligaste värdet.

```text
{missing_values.to_string()}
```

## 6. Val av X och y

Modellens **y-värde** är:

```text
{TARGET}
```

Numeriska X-värden:

```text
{", ".join(numeric_features)}
```

Kategoriska X-värden:

```text
{", ".join(categorical_features)}
```

Kolumner som inte används som features:

- adress
- käll-URL och detalj-URL
- exakt säljdatum
- kr/kvm
- prisförändring från utgångspris till slutpris
- helt tomma metadatafält
- datainsamlingsflaggor
- outlier-flaggor

Dessa tas bort för att minska risken för brus, överanpassning och dataläckage.

## 7. Feature engineering

För att hjälpa modellen har några extra features skapats:

- `total_area_m2` = boarea + biarea
- `has_asking_price` = om bostaden har känt/beräknat utgångspris
- `has_extra_area` = om biarea finns
- `has_plot_area` = om tomtarea finns
- `quarter` = kvartal baserat på månad
- `area_rooms_interaction` = boarea × antal rum

Dessa features bygger inte på slutpriset och kan därför användas utan att ge
dataläckage.

## 8. Modeller

Följande modeller testas:

1. Linear Regression med log-transformerad target
2. Decision Tree Regressor med log-transformerad target
3. Random Forest Regressor med log-transformerad target
4. Gradient Boosting Regressor med log-transformerad target

Log-transformeringen används eftersom bostadspriser varierar mycket i nivå.
Det gör att modellen i högre grad optimerar relativa prisskillnader.

## 9. Resultat

{results_table}

Den bästa modellen blev **{best_model_name}** baserat på lägst RMSE.

### Tolkning

- **MAE** visar hur många kronor modellen i genomsnitt missar.
- **RMSE** straffar stora fel hårdare.
- **R²** visar hur mycket av variationen i slutpris som modellen förklarar.
- **MAPE** visar genomsnittligt procentuellt fel.

För den bästa modellen är MAPE **{best_row["MAPE"]:.1f} %**. Det betyder att
modellen i genomsnitt missar slutpriset med ungefär denna procentandel.

## 10. Figurer

Figurerna sparas i `outputs/figures/`.

Den tidigare stora korrelationsgrafen med alla numeriska kolumner har ersatts
med en tydligare graf som bara visar varje variabels korrelation med slutpris.
Det gör det enklare att se vilka variabler som har starkast samband med
bostadspriset.

- `01_price_distribution.png` – fördelning av slutpriser
- `02_living_area_vs_price.png` – boarea jämfört med slutpris
- `03_correlation_with_price.png` – visar vilka numeriska variabler som har starkast samband med slutpris
- `04_actual_vs_predicted.png` – faktiskt vs predikterat slutpris
- `05_residuals.png` – modellens fel
- `06_asking_price_vs_final_price.png` – utgångspris jämfört med slutpris
- `07_price_by_property_type.png` – slutpris per bostadstyp
- `08_price_change_distribution.png` – prisförändring från utgångspris

Alla prisgrafer visas i **miljoner SEK**. Exempel: 6,5 betyder 6 500 000 kr.

## 11. Slutsats

Resultatet visar att det går att skapa en användbar modell för bostadspriser i
Partille kommun, men också att modellen har tydliga begränsningar.

Mer data hjälper bara delvis. För att få en klart bättre modell behövs framför
allt bättre information om varje bostad, exempelvis skick, renoveringsstandard,
energiklass och mer exakt geografisk information.

Modellen kan därför användas som ett analytiskt stöd, men inte som ensam grund
för köp, försäljning eller värdering.

## 12. Etisk reflektion

En bostadsprismodell påverkar ekonomiska beslut och bör därför användas med
försiktighet. Om datan är ofullständig kan modellen ge missvisande resultat för
vissa områden eller bostadstyper.

Resultatet ska därför ses som ett beslutsstöd, inte som ett facit. En mänsklig
bedömning bör alltid finnas kvar.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ensure_output_dirs()

    df = load_data()
    df = add_engineered_features(df)

    save_basic_visualizations(df)

    feature_columns = get_feature_columns(df)
    X = df[feature_columns]
    y = df[TARGET]

    print("\nKontroll av modellens indata")
    print("---------------------------")
    print(f"Target/y: {TARGET}")
    print(f"Antal features/X: {len(feature_columns)}")
    print("Feature-kolumner/X:")
    for column in feature_columns:
        print(f"  - {column}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(df)
    models = create_models(preprocessor)

    scores = {}
    fitted_models = {}

    for model_name, model in models.items():
        print(f"Tränar modell: {model_name}")
        model.fit(X_train, y_train)
        scores[model_name] = evaluate_model(model, X_test, y_test)
        fitted_models[model_name] = model

    results = pd.DataFrame(scores).T.sort_values("RMSE")
    best_model_name = results.index[0]
    best_model = fitted_models[best_model_name]

    save_prediction_plots(best_model, X_test, y_test)

    joblib.dump(best_model, MODEL_DIR / "best_housing_price_model.joblib")

    write_report(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        results=results,
        best_model_name=best_model_name,
    )

    print("\nProjektet är klart.")
    print(f"Datafil: {DATA_PATH}")
    print(f"Target/y: {TARGET}")
    print(f"Antal rader efter städning: {len(df)}")
    print(f"Bästa modell: {best_model_name}")
    print(results.round(3).to_string())
    print("\nTolkning:")
    print("MAPE visar genomsnittligt procentuellt fel.")
    print("Priser i graferna visas i miljoner SEK. Exempel: 6,5 betyder 6 500 000 kr.")
    print(f"Rapport: {REPORT_PATH}")
    print(f"Figurer: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
