
# =====================================================================
# RAPPORT
# =====================================================================
import pandas as pd

from modules.config import DATA_PATH, PROJECT_ROOT, REPORT_PATH, TARGET, USE_ASKING_PRICE_AS_FEATURE
from modules.functions import format_msek, format_sek

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
    """Skriv den slutliga Markdown-rapporten till aktuell rapportfil."""
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

Flera regressionsmodeller jämförs för att se vilken metod som bäst fångar
sambanden i bostadsdatan. Linear Regression används som en enkel och tydlig
basmodell, medan Decision Tree, Random Forest och Gradient Boosting kan fånga
mer komplexa och icke-linjära samband mellan bostädernas egenskaper och
slutpris. Slutpriset log-transformeras under träningen för att modellen inte
ska påverkas lika mycket av de dyraste bostäderna.

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
