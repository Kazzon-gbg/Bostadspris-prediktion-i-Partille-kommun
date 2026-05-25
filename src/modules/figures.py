
# -----------------------------------------------------------------
# FIGURER
# -----------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modules.config import FIGURE_DIR, TARGET
from modules.functions import add_price_columns_for_plots

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
        
    # Tydligare korrelationsgraf:
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
