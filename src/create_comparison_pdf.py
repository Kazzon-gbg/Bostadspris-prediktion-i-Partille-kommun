"""
Skapa en Markdown- och PDF-rapport som jämför 1, 2 och 3 års data.

Exempel:
    .venv/bin/python src/create_comparison_pdf.py
"""
import os
from datetime import datetime
from textwrap import wrap

from modules.config import PROJECT_ROOT


os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M")
COMPARISON_MD_PATH = PROJECT_ROOT / "reports" / f"report_jamforelse_{TIMESTAMP}.md"
COMPARISON_PDF_PATH = PROJECT_ROOT / "reports" / f"report_jamforelse_{TIMESTAMP}.pdf"

SUMMARY_ROWS = [
    {
        "Period": "1 år",
        "Dataset": "2025-05-19 till 2026-05-18",
        "Rader": 295,
        "Bästa modell": "Random Forest log-target",
        "MAE": 770235,
        "RMSE": 1102927,
        "R2": 0.553,
        "MAPE": 12.4,
    },
    {
        "Period": "2 år",
        "Dataset": "2024-05-20 till 2026-05-18",
        "Rader": 599,
        "Bästa modell": "Gradient Boosting log-target",
        "MAE": 710711,
        "RMSE": 1030689,
        "R2": 0.689,
        "MAPE": 11.3,
    },
    {
        "Period": "3 år",
        "Dataset": "2023-01-02 till 2026-05-18",
        "Rader": 948,
        "Bästa modell": "Random Forest log-target",
        "MAE": 615788,
        "RMSE": 968072,
        "R2": 0.732,
        "MAPE": 10.3,
    },
]

MODEL_ROWS = {
    "1 år": [
        ["Random Forest log-target", 770235, 1102927, 0.553, 12.4],
        ["Linear Regression log-target", 777273, 1114692, 0.543, 11.7],
        ["Gradient Boosting log-target", 829196, 1177665, 0.490, 12.6],
        ["Decision Tree log-target", 964996, 1258845, 0.417, 15.8],
    ],
    "2 år": [
        ["Gradient Boosting log-target", 710711, 1030689, 0.689, 11.3],
        ["Random Forest log-target", 700742, 1054690, 0.674, 11.0],
        ["Linear Regression log-target", 816522, 1137669, 0.621, 13.1],
        ["Decision Tree log-target", 870740, 1221806, 0.563, 13.6],
    ],
    "3 år": [
        ["Random Forest log-target", 615788, 968072, 0.732, 10.3],
        ["Gradient Boosting log-target", 619088, 975959, 0.727, 10.3],
        ["Linear Regression log-target", 681125, 1050154, 0.684, 11.6],
        ["Decision Tree log-target", 737498, 1120247, 0.641, 12.0],
    ],
}


def _format_number(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _wrap(text: str, width: int = 92) -> str:
    return "\n".join(wrap(text, width=width))


def _build_markdown_report() -> str:
    """Bygg en tidsstämplad Markdown-rapport för jämförelsen."""
    summary_lines = [
        "| Period | Bästa modell enligt RMSE | Rader | MAE | RMSE | R² | MAPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in SUMMARY_ROWS:
        summary_lines.append(
            "| "
            f"{row['Period']} | {row['Bästa modell']} | {row['Rader']} | "
            f"{_format_number(row['MAE'])} kr | {_format_number(row['RMSE'])} kr | "
            f"{row['R2']:.3f} | {row['MAPE']:.1f} % |"
        )

    model_sections = []
    for period, rows in MODEL_ROWS.items():
        model_lines = [
            f"### {period}",
            "",
            "| Modell | MAE | RMSE | R² | MAPE |",
            "|---|---:|---:|---:|---:|",
        ]
        for model_name, mae, rmse, r2, mape in rows:
            model_lines.append(
                f"| {model_name} | {_format_number(mae)} kr | {_format_number(rmse)} kr | "
                f"{r2:.3f} | {mape:.1f} % |"
            )
        model_sections.append("\n".join(model_lines))

    return f"""# Slutrapport: jämförelse mellan 1, 2 och 3 år

Skapad: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Syfte

Denna slutrapport jämför tre separata modellrapporter:

- Rapport 1: modell tränad på senaste 1 året
- Rapport 2: modell tränad på senaste 2 åren
- Rapport 3: modell tränad på hela datasetet, cirka 3 år

Syftet är att undersöka om bostadsprismodellen blir bättre av mer historisk
data eller av kortare och mer aktuell data.

## Sammanfattning

{chr(10).join(summary_lines)}

Resultatet visar att 3-årsmodellen presterar bäst i denna jämförelse. Den har
lägst MAE, lägst RMSE, högst R² och lägst MAPE.

## Modelljämförelser per period

{chr(10).join(model_sections)}

## Tolkning

1-årsmodellen är mest aktuell men har bara 295 rader. Den blir därför mer
känslig för vilka bostäder som hamnar i tränings- och testdata.

2-årsmodellen är en tydlig förbättring jämfört med 1-årsmodellen. Den använder
599 rader och fångar fler variationer mellan bostadstyper, områden och
prisnivåer.

3-årsmodellen använder 948 rader och ger bäst total prestanda. I detta projekt
väger den större datamängden tyngre än fördelen med att bara använda den mest
aktuella datan.

## Slutsats

Huvudmodellen bör vara 3-årsmodellen. Den ger bäst balans mellan stabilitet,
förklaringsgrad och prediktionsfel.

Samtidigt är 1- och 2-årsrapporterna viktiga eftersom de visar att valet av
dataperiod har testats praktiskt. Det stärker projektets analys och visar att
modellens resultat inte bara accepteras utan jämförs och tolkas.
"""


def _add_footer(fig: plt.Figure, page_number: int) -> None:
    fig.text(
        0.5,
        0.025,
        f"Jämförelse 1, 2 och 3 år | sida {page_number}",
        ha="center",
        fontsize=7.5,
        color="#6b7280",
    )


def _add_table(
    ax: plt.Axes,
    rows: list[list[str]],
    columns: list[str],
    bbox: list[float],
    font_size: float,
    col_widths: list[float] | None = None,
) -> None:
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
        colWidths=col_widths,
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


def _save_summary_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.0, 0.90), 1.0, 0.10, transform=ax.transAxes, facecolor="#203040", edgecolor="none"))
    ax.text(0.025, 0.962, "Jämförelse av träningsperioder", fontsize=18, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.025, 0.925, "Bostadspris-prediktion i Partille kommun", fontsize=10.5, color="#dbe5ee", transform=ax.transAxes)
    ax.text(0.73, 0.943, f"Skapad {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=8.8, color="#dbe5ee", transform=ax.transAxes)

    intro = (
        "Rapporten jämför samma maskininlärningsflöde tränat på 1, 2 och 3 års bostadsdata. "
        "Målet är att undersöka balansen mellan aktuell data och tillräckligt många observationer. "
        "Bästa modell väljs enligt lägst RMSE, eftersom stora fel i bostadspris är särskilt viktiga."
    )
    ax.text(0.04, 0.86, _wrap(intro), fontsize=10, va="top", linespacing=1.35, color="#25313b", transform=ax.transAxes)

    summary_table = []
    for row in SUMMARY_ROWS:
        summary_table.append([
            row["Period"],
            str(row["Rader"]),
            row["Bästa modell"].replace(" log-target", ""),
            _format_number(row["MAE"]),
            _format_number(row["RMSE"]),
            f"{row['R2']:.3f}",
            f"{row['MAPE']:.1f} %",
        ])
    _add_table(
        ax,
        summary_table,
        ["Period", "Rader", "Bästa modell", "MAE", "RMSE", "R2", "MAPE"],
        bbox=[0.04, 0.61, 0.92, 0.16],
        font_size=7.2,
        col_widths=[0.11, 0.10, 0.30, 0.13, 0.14, 0.10, 0.12],
    )

    ax.text(0.04, 0.54, "Huvudslutsats", fontsize=13, fontweight="bold", color="#1f2933", transform=ax.transAxes)
    conclusion = (
        "3-årsmodellen är bäst i denna jämförelse. Den har lägst genomsnittligt fel, lägst RMSE, "
        "högst R2 och lägst procentuellt fel. Det visar att den större datamängden i detta projekt "
        "väger tyngre än fördelen med en kortare och mer aktuell period."
    )
    ax.text(0.04, 0.51, _wrap(conclusion), fontsize=10, va="top", linespacing=1.35, color="#25313b", transform=ax.transAxes)

    ax.text(0.04, 0.39, "Tolkning", fontsize=13, fontweight="bold", color="#1f2933", transform=ax.transAxes)
    interpretation = (
        "1-årsmodellen är mest aktuell men har bara 295 rader och blir därför mer känslig. "
        "2-årsmodellen är en tydlig förbättring med 599 rader. 3-årsmodellen har 948 rader "
        "och bäst täckning av olika bostadstyper, områden och prisnivåer."
    )
    ax.text(0.04, 0.36, _wrap(interpretation), fontsize=10, va="top", linespacing=1.35, color="#25313b", transform=ax.transAxes)

    _add_footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def _save_metric_page(pdf: PdfPages) -> None:
    df = pd.DataFrame(SUMMARY_ROWS)
    periods = df["Period"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.suptitle("Resultat per tidsperiod", fontsize=17, fontweight="bold", color="#1f2933", y=0.955)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.18, hspace=0.40, wspace=0.32)

    chart_specs = [
        ("MAE", "MAE, MSEK", "Genomsnittligt absolut fel"),
        ("RMSE", "RMSE, MSEK", "Stora fel straffas hårdare"),
        ("R2", "R2", "Förklarad variation"),
        ("MAPE", "MAPE, procent", "Genomsnittligt procentuellt fel"),
    ]

    for ax, (column, ylabel, title) in zip(axes.flatten(), chart_specs, strict=True):
        values = df[column].tolist()
        ax.plot(periods, values, marker="o", linewidth=2.2, color="#2563eb")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        value_min = min(values)
        value_max = max(values)
        value_range = value_max - value_min if value_max != value_min else abs(value_max) * 0.08
        y_padding = value_range * 0.22
        ax.set_ylim(value_min - y_padding, value_max + y_padding)
        if column in {"MAE", "RMSE"}:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1_000_000:.2f}"))

    fig.text(
        0.08,
        0.105,
        _wrap("Diagrammen visar att 3-årsmodellen har bäst resultat. Felet sjunker stegvis från 1 år till 2 år och vidare till 3 år, samtidigt som R2 ökar."),
        fontsize=9.5,
        color="#25313b",
    )
    _add_footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def _save_model_tables_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.04, 0.955, "Modelljämförelse inom varje period", fontsize=17, fontweight="bold", color="#1f2933", transform=ax.transAxes)

    y_positions = [0.66, 0.37, 0.08]
    for period, y_position in zip(["1 år", "2 år", "3 år"], y_positions, strict=True):
        ax.text(0.04, y_position + 0.22, period, fontsize=12, fontweight="bold", color="#1f2933", transform=ax.transAxes)
        rows = [
            [
                model_name.replace(" log-target", ""),
                _format_number(mae),
                _format_number(rmse),
                f"{r2:.3f}",
                f"{mape:.1f} %",
            ]
            for model_name, mae, rmse, r2, mape in MODEL_ROWS[period]
        ]
        _add_table(
            ax,
            rows,
            ["Modell", "MAE", "RMSE", "R2", "MAPE"],
            bbox=[0.04, y_position, 0.92, 0.18],
            font_size=7.6,
            col_widths=[0.36, 0.18, 0.18, 0.12, 0.16],
        )

    _add_footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    COMPARISON_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_MD_PATH.write_text(_build_markdown_report(), encoding="utf-8")

    with PdfPages(COMPARISON_PDF_PATH) as pdf:
        _save_summary_page(pdf)
        _save_metric_page(pdf)
        _save_model_tables_page(pdf)
    print(f"Skapade Markdown-rapport: {COMPARISON_MD_PATH}")
    print(f"Skapade PDF-rapport: {COMPARISON_PDF_PATH}")


if __name__ == "__main__":
    main()
