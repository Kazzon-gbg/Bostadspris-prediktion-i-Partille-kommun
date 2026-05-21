"""
Kör hela modellflödet och skapa en tidsstämplad PDF-rapport i reports/.

Exempel:
    .venv/bin/python src/train_model_partille_toPDF.py
"""
import os

from modules.config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import joblib
import matplotlib

matplotlib.use("Agg")

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
    pdf_path = write_pdf_report(
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
    print(f"\nMarkdown-rapport: {REPORT_PATH}")
    print(f"PDF-rapport: {pdf_path}")
    print(f"Figurer: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
