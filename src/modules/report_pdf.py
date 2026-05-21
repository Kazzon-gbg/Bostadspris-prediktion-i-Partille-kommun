# =====================================================================
# PDF-RAPPORT
# =====================================================================
from datetime import datetime
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from modules.config import DATA_PATH, FIGURE_DIR, PROJECT_ROOT, REPORT_PATH, TARGET, USE_ASKING_PRICE_AS_FEATURE
from modules.functions import format_msek, format_sek
from modules.report import get_dataset_summary


FIGURE_CAPTIONS = [
    ("01_price_distribution.png", "Fördelning av slutpriser", "Visar hur slutpriserna är spridda i datasetet. Diagrammet gör det lättare att se om de flesta bostäder ligger i ett visst prisintervall och om det finns ovanligt dyra objekt."),
    ("02_living_area_vs_price.png", "Slutpris jämfört med boarea", "Visar sambandet mellan bostadens storlek och slutpris. Punkterna färgas efter bostadstyp när den informationen finns."),
    ("03_correlation_with_price.png", "Korrelation med slutpris", "Visar vilka numeriska variabler som har starkast positivt eller negativt samband med slutpriset."),
    ("04_actual_vs_predicted.png", "Faktiskt jämfört med predikterat slutpris", "Visar hur nära modellens prediktioner ligger de faktiska slutpriserna. Punkter nära den röda linjen betyder bättre träff."),
    ("05_residuals.png", "Residualer", "Visar modellens fel, alltså faktiskt slutpris minus predikterat slutpris. En bra modell har många fel nära noll."),
    ("06_asking_price_vs_final_price.png", "Utgångspris jämfört med slutpris", "Visar hur utgångspris och slutpris förhåller sig till varandra i datasetet."),
    ("07_price_by_property_type.png", "Slutpris per bostadstyp", "Jämför prisnivåer mellan olika bostadstyper och visar spridningen inom varje grupp."),
    ("08_price_change_distribution.png", "Prisförändring från utgångspris", "Visar hur mycket slutpriset skiljer sig från utgångspriset när den informationen finns."),
]


def get_timestamped_pdf_path() -> object:
    """Skapa filnamn enligt report_YYMMDD_HHMM.pdf i reports-mappen."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    return REPORT_PATH.parent / f"report_{timestamp}.pdf"


def _add_footer(fig: plt.Figure, page_number: int) -> None:
    fig.text(0.5, 0.025, f"Bostadspris-prediktion i Partille kommun | sida {page_number}", ha="center", fontsize=7.5, color="#6b7280")


def _wrapped_text(text: str, width: int = 95) -> str:
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(wrap(paragraph, width=width))
    return "\n".join(lines)


def _feature_summary(features: list[str], max_items: int = 10) -> str:
    """Sammanfatta långa feature-listor så de får plats i PDF-layouten."""
    if len(features) <= max_items:
        return ", ".join(features)

    visible_features = ", ".join(features[:max_items])
    remaining_count = len(features) - max_items
    return f"{visible_features} samt {remaining_count} till"


def _read_cropped_image(image_path: object) -> Image.Image:
    """Beskär vit marginal runt sparade diagram så de fyller PDF-ytan bättre."""
    image = Image.open(image_path).convert("RGB")
    pixels = image.load()
    width, height = image.size

    left, upper = width, height
    right, lower = 0, 0
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            if red < 245 or green < 245 or blue < 245:
                left = min(left, x)
                upper = min(upper, y)
                right = max(right, x)
                lower = max(lower, y)

    if right <= left or lower <= upper:
        return image

    padding = 18
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(width, right + padding)
    lower = min(height, lower + padding)
    return image.crop((left, upper, right, lower))


def _save_text_page(pdf: PdfPages, title: str, body: str, page_number: int) -> None:
    """Spara en enkel textsida i PDF-rapporten."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=20, fontweight="bold", color="#1f2933")
    fig.text(0.08, 0.885, _wrapped_text(body), fontsize=10.5, va="top", linespacing=1.35, color="#222222")
    _add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def _save_results_page(pdf: PdfPages, results: pd.DataFrame, best_model_name: str, page_number: int) -> None:
    """Spara en fristående resultatsida.

    Funktionen finns kvar om rapporten senare behöver en separat resultatsida.
    Nuvarande PDF-layout visar resultatet på översiktssidan.
    """
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    best_row = results.loc[best_model_name]
    ax.text(0.02, 0.96, "Resultat och modelljämförelse", fontsize=20, fontweight="bold", color="#1f2933", transform=ax.transAxes)
    ax.text(
        0.02,
        0.895,
        (
            f"Bästa modell: {best_model_name}\n"
            f"MAE: {format_sek(best_row['MAE'])} ({format_msek(best_row['MAE'])})\n"
            f"RMSE: {format_sek(best_row['RMSE'])} ({format_msek(best_row['RMSE'])})\n"
            f"R2: {best_row['R2']:.3f}\n"
            f"MAPE: {best_row['MAPE']:.1f} %"
        ),
        fontsize=11,
        va="top",
        linespacing=1.45,
        color="#222222",
        transform=ax.transAxes,
    )

    table_data = []
    for model_name, row in results.iterrows():
        table_data.append([
            model_name,
            f"{row['MAE']:,.0f}".replace(",", " "),
            f"{row['RMSE']:,.0f}".replace(",", " "),
            f"{row['R2']:.3f}",
            f"{row['MAPE']:.1f} %",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Modell", "MAE, kr", "RMSE, kr", "R2", "MAPE"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.45, 0.96, 0.30],
        colWidths=[0.40, 0.18, 0.18, 0.10, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee3")
        if row_index == 0:
            cell.set_facecolor("#e8eef3")
            cell.set_text_props(fontweight="bold", color="#1f2933")
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else "#f7f9fb")

    explanation = (
        "MAE visar hur många kronor modellen i genomsnitt missar. RMSE straffar stora fel hårdare "
        "och används här för att välja bästa modell. R2 visar hur mycket av variationen i slutpris "
        "som modellen förklarar. MAPE visar genomsnittligt procentuellt fel och är ofta lätt att "
        "tolka när bostäderna ligger på olika prisnivåer."
    )
    ax.text(0.02, 0.36, _wrapped_text(explanation, width=92), fontsize=10.5, va="top", linespacing=1.35, transform=ax.transAxes)

    _add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def _build_results_table_data(results: pd.DataFrame) -> list[list[str]]:
    """Formatera modellresultat som tabellrader för PDF-layouten."""
    table_data = []
    for model_name, row in results.iterrows():
        table_data.append([
            model_name,
            f"{row['MAE']:,.0f}".replace(",", " "),
            f"{row['RMSE']:,.0f}".replace(",", " "),
            f"{row['R2']:.3f}",
            f"{row['MAPE']:.1f} %",
        ])
    return table_data


def _add_results_table(ax: plt.Axes, results: pd.DataFrame, bbox: list[float], font_size: float = 7.8) -> None:
    """Lägg till modelljämförelsen som en tabell i en Matplotlib-yta."""
    table = ax.table(
        cellText=_build_results_table_data(results),
        colLabels=["Modell", "MAE, kr", "RMSE, kr", "R2", "MAPE"],
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
        colWidths=[0.40, 0.18, 0.18, 0.10, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee3")
        if row_index == 0:
            cell.set_facecolor("#e8eef3")
            cell.set_text_props(fontweight="bold", color="#1f2933")
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else "#f7f9fb")


def _save_overview_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    results: pd.DataFrame,
    best_model_name: str,
    page_number: int,
) -> None:
    """Spara PDF-rapportens första sida med sammanfattning och modellresultat."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    summary = get_dataset_summary(df)
    best_row = results.loc[best_model_name]
    asking_price_text = "används som feature" if USE_ASKING_PRICE_AS_FEATURE else "används inte som feature"

    ax.add_patch(plt.Rectangle((0.0, 0.90), 1.0, 0.10, transform=ax.transAxes, facecolor="#203040", edgecolor="none"))
    ax.text(0.025, 0.962, "Bostadspris-prediktion", fontsize=19, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.025, 0.925, "Partille kommun", fontsize=11, color="#dbe5ee", transform=ax.transAxes)
    ax.text(0.73, 0.943, f"Skapad {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=8.8, color="#dbe5ee", transform=ax.transAxes)

    intro = (
        f"Datasetet omfattar bostadsförsäljningar från {summary['min_date']} till {summary['max_date']}. "
        f"Efter städning används {summary['rows']} rader. Syftet är att prediktera slutpris i kronor "
        f"och jämföra flera regressionsmodeller med samma feature-underlag."
    )
    ax.text(0.025, 0.865, _wrapped_text(intro, width=92), fontsize=10, va="top", linespacing=1.35, color="#25313b", transform=ax.transAxes)

    metrics = [
        ("Bästa modell", best_model_name),
        ("MAE", f"{format_sek(best_row['MAE'])} ({format_msek(best_row['MAE'])})"),
        ("RMSE", f"{format_sek(best_row['RMSE'])} ({format_msek(best_row['RMSE'])})"),
        ("R2", f"{best_row['R2']:.3f}"),
        ("MAPE", f"{best_row['MAPE']:.1f} %"),
    ]
    card_positions = [
        (0.025, 0.735, 0.46, 0.075),
        (0.515, 0.735, 0.46, 0.075),
        (0.025, 0.635, 0.46, 0.075),
        (0.515, 0.635, 0.215, 0.075),
        (0.760, 0.635, 0.215, 0.075),
    ]
    for (label, value), (x, y, width, height) in zip(metrics, card_positions, strict=True):
        ax.add_patch(plt.Rectangle((x, y), width, height, transform=ax.transAxes, facecolor="#f4f7f9", edgecolor="#d7dee5", linewidth=0.8))
        ax.text(x + 0.015, y + height - 0.027, label, fontsize=8.4, color="#5f6b76", transform=ax.transAxes)
        value_font_size = 8.3 if label in {"MAE", "RMSE"} else 9.2
        ax.text(x + 0.015, y + 0.020, value, fontsize=value_font_size, fontweight="bold", color="#1f2933", transform=ax.transAxes)

    ax.text(0.025, 0.585, "Modelljämförelse", fontsize=12, fontweight="bold", color="#1f2933", transform=ax.transAxes)
    _add_results_table(ax, results, bbox=[0.025, 0.365, 0.95, 0.18], font_size=7.3)

    dataset_text = (
        f"Datafil: {DATA_PATH.relative_to(PROJECT_ROOT)}\n"
        f"Target/y: {TARGET}\n"
        f"Bostadstyper: {summary['property_types']}\n"
        f"Utgångspris i modellen: {asking_price_text}; rader med utgångspris: {summary['asking_price_count']}\n"
        f"Prisnivåer: lägst {format_sek(summary['min_price'])}, median {format_sek(summary['median_price'])}, "
        f"genomsnitt {format_sek(summary['mean_price'])}, högst {format_sek(summary['max_price'])}.\n\n"
        f"Numeriska features ({len(numeric_features)}): {_feature_summary(numeric_features)}\n"
        f"Kategoriska features ({len(categorical_features)}): {_feature_summary(categorical_features)}"
    )
    ax.text(0.025, 0.315, "Dataset och features", fontsize=12, fontweight="bold", color="#1f2933", transform=ax.transAxes)
    ax.text(0.025, 0.29, _wrapped_text(dataset_text, width=105), fontsize=7.7, va="top", linespacing=1.25, color="#25313b", transform=ax.transAxes)

    interpretation = (
        "Tolkning: MAE visar genomsnittligt fel i kronor, RMSE straffar stora fel hårdare, "
        "R2 visar förklarad variation och MAPE visar genomsnittligt procentuellt fel. "
        "Modellen är ett analytiskt stöd; faktorer som skick, mikroläge, planlösning, ränteläge "
        "och budgivningsläge saknas ofta i datasetet."
    )
    ax.add_patch(plt.Rectangle((0.025, 0.055), 0.95, 0.085, transform=ax.transAxes, facecolor="#fbf7ec", edgecolor="#e4d8ba", linewidth=0.8))
    ax.text(0.045, 0.115, "Kort tolkning", fontsize=9.3, fontweight="bold", color="#604b1f", transform=ax.transAxes)
    ax.text(0.045, 0.094, _wrapped_text(interpretation, width=104), fontsize=7.7, va="top", linespacing=1.2, color="#3f3a2d", transform=ax.transAxes)

    _add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def _save_figure_grid_page(pdf: PdfPages, figure_items: list[tuple[str, str, str]], page_number: int) -> None:
    """Spara en A4-sida med upp till två diagram och förklarande bildtexter."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.955, "Diagram och modellkontroll", fontsize=16.5, fontweight="bold", color="#1f2933")
    fig.text(0.07, 0.932, "Varje diagram visar datakvalitet, prisbild eller modellens träffsäkerhet.", fontsize=8.8, color="#5f6b76")

    slots = [
        (0.07, 0.515, 0.86, 0.35),
        (0.07, 0.105, 0.86, 0.35),
    ]

    for (filename, title, caption), (x, y, width, height) in zip(figure_items, slots, strict=False):
        image_path = FIGURE_DIR / filename
        if not image_path.exists():
            continue

        panel = plt.Rectangle(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            facecolor="#f8fafc",
            edgecolor="#d8e0e7",
            linewidth=0.8,
            zorder=-10,
        )
        fig.add_artist(panel)
        fig.text(x + 0.018, y + height - 0.035, title, fontsize=11.0, fontweight="bold", color="#1f2933", va="top", zorder=10)
        fig.text(
            x + 0.018,
            y + height - 0.075,
            _wrapped_text(caption, width=31),
            fontsize=6.7,
            color="#3f4952",
            linespacing=1.18,
            va="top",
            zorder=10,
        )

        image = _read_cropped_image(image_path)
        ax = fig.add_axes([x + 0.235, y + 0.030, width - 0.265, height - 0.060])
        ax.set_zorder(5)
        ax.set_facecolor("white")
        ax.imshow(image)
        ax.axis("off")

    _add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf_report(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    results: pd.DataFrame,
    best_model_name: str,
) -> object:
    """Skapa en kompakt, förklarande PDF-rapport med resultat och figurer."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = get_timestamped_pdf_path()

    page_number = 1
    with PdfPages(pdf_path) as pdf:
        _save_overview_page(pdf, df, numeric_features, categorical_features, results, best_model_name, page_number)
        page_number += 1

        existing_figures = [
            figure_item
            for figure_item in FIGURE_CAPTIONS
            if (FIGURE_DIR / figure_item[0]).exists()
        ]
        for start_index in range(0, len(existing_figures), 2):
            _save_figure_grid_page(pdf, existing_figures[start_index:start_index + 2], page_number)
            page_number += 1

        if len(existing_figures) == 0:
            _save_text_page(
                pdf,
                "Diagram saknas",
                "Inga sparade diagram hittades i outputs/figures när PDF-rapporten skapades.",
                page_number,
            )
            page_number += 1

    return pdf_path
