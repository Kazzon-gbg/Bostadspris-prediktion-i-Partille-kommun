# =====================================================================
# HJÄLPFUNKTIONER
# =====================================================================
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from modules.config import (
    DATA_PATH,
    EXCLUDE_FEATURE_COLUMNS,
    FIGURE_DIR,
    MAX_VALID_FINAL_PRICE_SEK,
    MIN_VALID_FINAL_PRICE_SEK,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_PATH,
    TARGET,
    USE_ASKING_PRICE_AS_FEATURE,
)

def format_sek(value: float) -> str:
    """Formatera kronor så rapporten blir lättläst."""
    return f"{value:,.0f} kr".replace(",", " ")


def format_msek(value: float) -> str:
    """Formatera kronor som miljoner SEK."""
    return f"{value / 1_000_000:.2f} MSEK"


def ensure_output_dirs() -> None:
    """Skapa mapparna där skriptet sparar rapport, figurer och modellfil."""
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
    Skapa extra features som kan förbättra modellen.

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

    # Enkel interaktion: boarea multiplicerat med antal rum.
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
