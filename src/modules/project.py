# =====================================================================
# PROJEKTFLÖDE
# =====================================================================
from dataclasses import dataclass
""" Enklare att skapa klasser"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.config import DATA_PATH, FIGURE_DIR, MODEL_DIR, RANDOM_STATE, REPORT_PATH, TARGET
from modules.figures import save_basic_visualizations, save_prediction_plots
from modules.functions import (
    add_engineered_features,
    build_preprocessor,
    create_models,
    ensure_output_dirs,
    evaluate_model,
    get_feature_columns,
    load_data,
)
from modules.report import write_report
from modules.report_pdf import write_pdf_report


@dataclass
class HousingPriceProjectResult:
    """Resultatobjekt som samlar de viktigaste delarna från en projektkörning."""

    df: pd.DataFrame
    results: pd.DataFrame
    best_model_name: str
    numeric_features: list[str]
    categorical_features: list[str]
    pdf_path: Path | None = None


class HousingPriceProject:
    """
    Objektorienterad samordnare för hela maskininlärningsflödet.

    Klassen visar hur projektets delar hör ihop: data laddas, features skapas,
    modeller tränas, resultat utvärderas och rapporter sparas. Varje steg ligger
    i en egen metod så flödet blir lättare att felsöka, testa och vidareutveckla.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.df: pd.DataFrame | None = None
        self.feature_columns: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.fitted_models: dict[str, object] = {}
        self.results: pd.DataFrame | None = None
        self.best_model_name: str | None = None

    def prepare_data(self) -> pd.DataFrame:
        """Ladda data, skapa features och spara grundläggande visualiseringar."""
        ensure_output_dirs()
        self.df = add_engineered_features(load_data())
        save_basic_visualizations(self.df)
        self.feature_columns = get_feature_columns(self.df)
        return self.df

    def print_input_summary(self) -> None:
        """Skriv ut en enkel kontroll av X- och y-värden."""
        print("\nKontroll av modellens indata")
        print("---------------------------")
        print(f"Target/y: {TARGET}")
        print(f"Antal features/X: {len(self.feature_columns)}")
        print("Feature-kolumner/X:")
        for column in self.feature_columns:
            print(f"  - {column}")

    def train_and_evaluate(self) -> pd.DataFrame:
        """Träna modellerna, utvärdera dem och spara bästa modellen."""
        if self.df is None:
            raise RuntimeError("Data måste förberedas innan modellerna tränas.")

        X = self.df[self.feature_columns]
        y = self.df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        preprocessor, self.numeric_features, self.categorical_features = build_preprocessor(self.df)
        models = create_models(preprocessor)

        scores = {}
        self.fitted_models = {}
        for model_name, model in models.items():
            print(f"Tränar modell: {model_name}")
            model.fit(X_train, y_train)
            scores[model_name] = evaluate_model(model, X_test, y_test)
            self.fitted_models[model_name] = model

        self.results = pd.DataFrame(scores).T.sort_values("RMSE")
        self.best_model_name = str(self.results.index[0])
        best_model = self.fitted_models[self.best_model_name]

        save_prediction_plots(best_model, X_test, y_test)
        joblib.dump(best_model, MODEL_DIR / "best_housing_price_model.joblib")
        return self.results

    def write_reports(self, create_pdf: bool = False) -> Path | None:
        """Skriv Markdown-rapport och, vid behov, även PDF-rapport."""
        if self.df is None or self.results is None or self.best_model_name is None:
            raise RuntimeError("Modellerna måste tränas innan rapporter kan skapas.")

        write_report(
            df=self.df,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            results=self.results,
            best_model_name=self.best_model_name,
        )

        if not create_pdf:
            return None

        return write_pdf_report(
            df=self.df,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            results=self.results,
            best_model_name=self.best_model_name,
        )

    def run(self, create_pdf: bool = False) -> HousingPriceProjectResult:
        """Kör hela projektet från CSV-fil till färdig modell och rapport."""
        df = self.prepare_data()
        self.print_input_summary()
        results = self.train_and_evaluate()
        pdf_path = self.write_reports(create_pdf=create_pdf)

        return HousingPriceProjectResult(
            df=df,
            results=results,
            best_model_name=str(self.best_model_name),
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            pdf_path=pdf_path,
        )

    def print_completion_summary(self, result: HousingPriceProjectResult) -> None:
        """Skriv ut slutstatus efter en lyckad körning."""
        print("\nProjektet är klart.")
        print(f"Datafil: {DATA_PATH}")
        print(f"Target/y: {TARGET}")
        print(f"Antal rader efter städning: {len(result.df)}")
        print(f"Bästa modell: {result.best_model_name}")
        print(result.results.round(3).to_string())
        print("\nTolkning:")
        print("MAPE visar genomsnittligt procentuellt fel.")
        print("Priser i graferna visas i miljoner SEK. Exempel: 6,5 betyder 6 500 000 kr.")
        print(f"Markdown-rapport: {REPORT_PATH}")
        if result.pdf_path is not None:
            print(f"PDF-rapport: {result.pdf_path}")
        print(f"Figurer: {FIGURE_DIR}")
