# Bostadspris-prediktion i Partille kommun med Python och AI

**Mikael Karlsson 2026**   

Detta projekt är ett examensarbete för kursen **ITHS Pythonprogrammering för
AI-utveckling VT-2026**

Projektet undersöker om det går att prediktera slutpris för bostäder i
**Partille kommun**, inklusive **Sävedalen**, med hjälp av verklig bostadsdata
och maskininlärning.

Projektet visar ett komplett AI-arbetsflöde:

1. läsa in bostadsdata från CSV
2. kontrollera och rensa data
3. skapa features
4. visualisera data
5. träna flera maskininlärningsmodeller
6. jämföra modellernas resultat
7. analysera om 1, 2 eller 3 års data fungerar bäst
8. skapa Markdown- och PDF-rapporter
9. reflektera över modellens begränsningar och etiska risker

Själva hämtningen av bostadsdata från Booli ingår **inte** i detta projekt.
Datahämtningen har gjorts separat och detta projekt fokuserar på analys,
modellering, rapportering och tolkning.

## Problemformulering

Kan vi prediktera bostäders slutpris baserat på egenskaper som:

- boarea
- biarea
- antal rum
- tomtarea
- bostadstyp
- område
- Sävedalen/ej Sävedalen
- år och månad för försäljning
- utgångspris
- ungefärliga geografiska features, till exempel avstånd till Partille centrum,
  Göteborg centrum och Sävedalen centrum

Eftersom slutpriset är ett numeriskt värde är detta ett **regressionsproblem**.

## Dataset

Projektet använder tre dataset som bygger på samma ursprungliga bostadsdata,
men med olika tidsperioder. Syftet är att testa om modellen blir bättre av mer
historisk data eller av kortare och mer aktuell data.

| Rapport | Period | Datafil | Antal rader efter städning | Syfte |
|---|---|---|---:|---|
| Rapport 1 | 1 år | `data/partille_housing_real_last_1_year.csv` | 295 | testar mest aktuell data |
| Rapport 2 | 2 år | `data/partille_housing_real_last_2_years.csv` | 599 | testar en kompromiss mellan aktualitet och datamängd |
| Rapport 3 | 3 år | `data/partille_housing_real_2023_today.csv` | 948 | huvudmodell med mest data |

Den ursprungliga huvudfilen är:

```text
data/partille_housing_real_2023_today.csv
```

Den består av verkliga bostadsförsäljningar i Partille kommun från
**2023-01-02 till 2026-05-18**.

Utifrån huvudfilen har två kortare CSV-filer skapats:

```text
data/partille_housing_real_last_1_year.csv
data/partille_housing_real_last_2_years.csv
```

1-årsfilen omfattar **2025-05-19 till 2026-05-18** och 2-årsfilen omfattar
**2024-05-20 till 2026-05-18**. Datumen startar vid första faktiska försäljning
som finns i respektive filtrerad period.

Target/y-värdet är:

```text
final_price_sek
```

Det är alltså slutpriset som modellen försöker prediktera.

Exempel på kolumner i datasetet:

```text
property_type
area_name
is_savedalen
sold_date
year
month
final_price_sek
asking_price_sek
living_area_m2
extra_area_m2
rooms
plot_area_m2
area_group
distance_to_partille_center_km
distance_to_gothenburg_center_km
distance_to_savedalen_center_km
```

## Viktigt om datan

Datasetet är verklig bostadsdata och innehåller därför saknade värden. Exempelvis
saknas ibland utgångspris, rum, biarea eller tomtarea.

I modellens preprocessing hanteras detta genom:

- medianimputering för numeriska kolumner
- imputering med vanligaste värde för kategoriska kolumner
- one-hot encoding för kategoriska kolumner

Vissa kolumner används inte som features eftersom de kan skapa brus,
överanpassning eller dataläckage. Exempel:

```text
address
source_url
detail_url
sold_date
price_per_m2
price_change_sek
price_change_percent
bid_change_percent
energy_class
operating_cost_sek_per_year
ownership_type
days_on_market
```

`price_per_m2` och prisförändringskolumnerna bygger helt eller delvis på
slutpriset och ska därför inte användas som input när modellen ska prediktera
slutpris. URL- och adresskolumner är identifierare och riskerar att göra
modellen för beroende av enskilda objekt i stället för generella mönster.

## Projektstruktur

```text
.
├── data/
│   ├── partille_housing_real_2023_today.csv
│   ├── partille_housing_real_last_1_year.csv
│   └── partille_housing_real_last_2_years.csv
├── src/
│   ├── create_comparison_pdf.py
│   ├── train_model_partille_1year.py
│   ├── train_model_partille_2years.py
│   ├── train_model_partille_3years.py
│   └── modules/
│       ├── config.py
│       ├── figures.py
│       ├── functions.py
│       ├── project.py
│       ├── report.py
│       └── report_pdf.py
├── reports/
│   ├── report_1year_YYMMDD_HHMM.md
│   ├── report_1year_YYMMDD_HHMM.pdf
│   ├── report_2years_YYMMDD_HHMM.md
│   ├── report_2years_YYMMDD_HHMM.pdf
│   ├── report_3years_YYMMDD_HHMM.md
│   ├── report_3years_YYMMDD_HHMM.pdf
│   ├── report_jamforelse_YYMMDD_HHMM.md
│   └── report_jamforelse_YYMMDD_HHMM.pdf
├── outputs/
│   ├── figures/
│   ├── figures_1year/
│   ├── figures_2years/
│   ├── models/
│   ├── models_1year/
│   └── models_2years/
├── requirements.txt
└── README.md
```

## Installation

Skapa en virtuell miljö och installera paketen:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Paket som används:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- pillow

## Kör projektet

Projektet har ett separat körskript för varje träningsperiod. Varje
modellskript skapar både Markdown-rapport och PDF-rapport.

```bash
python src/train_model_partille_1year.py
python src/train_model_partille_2years.py
python src/train_model_partille_3years.py
python src/create_comparison_pdf.py
```

De tre första skripten skapar varsin modellrapport. Det sista skriptet skapar
slutrapporten som jämför 1, 2 och 3 år.

## Rapporter

Alla rapporter sparas i `reports/` och följer samma namngivning:

```text
report_rapportnamn_YYMMDD_HHMM.md
report_rapportnamn_YYMMDD_HHMM.pdf
```

| Rapport | Kommando | Markdown | PDF |
|---|---|---|---|
| Rapport 1, 1 år | `python src/train_model_partille_1year.py` | `reports/report_1year_YYMMDD_HHMM.md` | `reports/report_1year_YYMMDD_HHMM.pdf` |
| Rapport 2, 2 år | `python src/train_model_partille_2years.py` | `reports/report_2years_YYMMDD_HHMM.md` | `reports/report_2years_YYMMDD_HHMM.pdf` |
| Rapport 3, 3 år | `python src/train_model_partille_3years.py` | `reports/report_3years_YYMMDD_HHMM.md` | `reports/report_3years_YYMMDD_HHMM.pdf` |
| Slutrapport, jämförelse | `python src/create_comparison_pdf.py` | `reports/report_jamforelse_YYMMDD_HHMM.md` | `reports/report_jamforelse_YYMMDD_HHMM.pdf` |

Rapport 1, 2 och 3 är separata modellrapporter. Slutrapporten är en jämförelse
mellan de tre modellrapporterna.

## Modellflöde

Varje modellkörning gör följande:

1. Läser in rätt CSV-fil.
2. Gör grundläggande datakontroll.
3. Skapar extra features.
4. Skapar visualiseringar.
5. Delar upp datan i träningsdata och testdata.
6. Tränar flera regressionsmodeller.
7. Utvärderar modellerna med MAE, RMSE, R² och MAPE.
8. Sparar bästa modellen.
9. Skapar Markdown-rapport.
10. Skapar PDF-rapport.

## Objektorienterad struktur

Projektets huvudflöde körs via klassen `HousingPriceProject` i
`src/modules/project.py`.

Projektet använder både funktioner och objektorienterad programmering.
Hjälpfunktionerna i `src/modules/functions.py`, `src/modules/figures.py` och
`src/modules/report.py` gör avgränsade delar av arbetet, medan
`HousingPriceProject` fungerar som en samordnande klass för hela
maskininlärningsflödet.

Klassen visar objektorienterade principer genom att:

- lagra projektets tillstånd i attribut, till exempel dataset, feature-listor,
  tränade modeller, resultat och bästa modell
- dela upp arbetsflödet i metoder som `prepare_data()`, `train_and_evaluate()`,
  `write_reports()` och `run()`
- använda resultatobjektet `HousingPriceProjectResult` för att samla information
  från en färdig körning
- göra flödet mer testbart och felsökningsbart eftersom varje steg kan köras och
  kontrolleras separat

## Feature engineering

För att hjälpa modellen skapas flera extra features:

- `total_area_m2` = boarea + biarea
- `has_asking_price` = om bostaden har känt eller beräknat utgångspris
- `has_extra_area` = om biarea finns
- `has_plot_area` = om tomtarea finns
- `quarter` = kvartal baserat på månad
- `area_rooms_interaction` = boarea × antal rum

Dessa features bygger inte på slutpriset och kan därför användas utan att skapa
dataläckage.

## Modeller

Projektet jämför flera regressionsmodeller:

- Linear Regression med log-transformerad target
- Decision Tree Regressor med log-transformerad target
- Random Forest Regressor med log-transformerad target
- Gradient Boosting Regressor med log-transformerad target

Log-transformering används eftersom bostadspriser varierar mycket i nivå. Det
gör att modellen i högre grad hanterar relativa prisskillnader och inte bara
domineras av de dyraste objekten.

## Utvärderingsmått

Modellerna utvärderas med:

- **MAE** – genomsnittligt absolut fel i kronor
- **RMSE** – rotmedelkvadratfel, straffar stora fel hårdare
- **R²** – hur stor del av variationen i slutpris modellen förklarar
- **MAPE** – genomsnittligt procentuellt fel

RMSE används för att välja bästa modell i rapporterna eftersom stora fel i
bostadspris kan vara särskilt viktiga. MAPE är pedagogiskt eftersom det visar
felet i procent i stället för bara kronor.

## Resultat

| Period | Bästa modell enligt RMSE | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| 1 år | Random Forest log-target | 770 235 kr | 1 102 927 kr | 0.553 | 12.4 % |
| 2 år | Gradient Boosting log-target | 710 711 kr | 1 030 689 kr | 0.689 | 11.3 % |
| 3 år | Random Forest log-target | 615 788 kr | 968 072 kr | 0.732 | 10.3 % |

Resultatet visar att 3-årsmodellen är bäst i detta projekt. Den har lägst MAE,
lägst RMSE, högst R² och lägst MAPE.

Tolkningen är att 1-årsdata är mest aktuell men innehåller för få rader för att
ge samma stabilitet. 2-årsmodellen är en tydlig förbättring jämfört med 1 år,
men 3 år ger fortfarande bäst resultat.

Denna del stärker projektet eftersom valet av dataperiod inte bara antas, utan
testas och analyseras.

## Visualiseringar

Skripten sparar figurer i respektive output-mapp:

```text
outputs/figures/
outputs/figures_1year/
outputs/figures_2years/
```

Exempel på figurer:

```text
01_price_distribution.png
02_living_area_vs_price.png
03_correlation_with_price.png
04_actual_vs_predicted.png
05_residuals.png
06_asking_price_vs_final_price.png
07_price_by_property_type.png
08_price_change_distribution.png
```

Alla prisgrafer visas i **miljoner SEK** för att vara lättare att läsa.
Exempel: `6,5` på en axel betyder cirka `6 500 000 kr`.

## Avgränsning

Det här projektet fokuserar på:

- dataförståelse
- datarensning inför modellering
- feature engineering
- visualisering
- modellträning
- modellutvärdering
- rapportering
- etisk reflektion

Själva insamlingen av data från Booli ingår inte i projektet. Den hanteras
separat och resultatet från den processen är CSV-filen:

```text
data/partille_housing_real_2023_today.csv
```

## Begränsningar

Även med verklig data från flera år är bostadspriser svåra att prediktera exakt.
Viktiga faktorer saknas fortfarande, till exempel:

- exakt mikroläge
- skick och renoveringsstandard
- pålitligt byggår för alla objekt
- energiklass
- planlösning
- utsikt
- närhet till skolor, service och kommunikationer
- budgivningsläge
- ränteläge och marknadsläge

Därför ska modellen ses som ett analytiskt stöd, inte som ett facit.

## Etisk reflektion

En bostadsprismodell kan påverka ekonomiska beslut som köp, försäljning,
lånelöften och uppfattningar om ett områdes värde. Därför är det viktigt att
modellen inte presenteras som ett objektivt facit. Den bygger på historisk data
och på de variabler som finns tillgängliga i datasetet.

Det finns flera etiska risker:

- **Ofullständig data:** Skick, renoveringsstandard, planlösning, utsikt,
  buller, närhet till service och andra kvalitativa faktorer saknas ofta.
- **Representationsproblem:** Om vissa områden eller bostadstyper har färre
  observationer kan modellen bli mindre träffsäker för just dessa grupper.
- **Marknadsförändringar:** Ränteläge, konjunktur och lokala marknadstrender kan
  förändras snabbt. En modell som tränats på historiska försäljningar kan därför
  ge missvisande resultat i ett nytt marknadsläge.
- **Övertro på modellen:** Ett exakt tal i kronor kan uppfattas som mer säkert
  än det egentligen är. Därför redovisas även felmått som MAE, RMSE, R² och
  MAPE.

För att minska risken för felaktig användning redovisar projektet både
modellens resultat och dess begränsningar. Modellen bör användas som ett
analytiskt stöd tillsammans med mänsklig bedömning, inte som ensam grund för
köp, försäljning, värdering eller lånebeslut.

## Framtida förbättringsförslag

- Komplettera datasetet med skick, renoveringsår, energiklass och driftskostnad
  där det går.
- Lägga till mer exakt lägesinformation, exempelvis avstånd till skola, service
  och kollektivtrafik.
- Validera modellen med cross-validation så resultatet blir mindre beroende av
  en enda train/test-split.
- Jämföra resultat med och utan `asking_price_sek`, eftersom utgångspris ofta är
  en stark men ibland saknad feature.
- Undersöka feature importance för bästa modellen och ta bort features som mest
  skapar brus.
- Lägga till marknadsfeatures, till exempel räntenivå, prisindex.

## Verktyg och programvara

- Python 3.14.5
- Pandas, NumPy, Matplotlib, Seaborn och Scikit-learn
- Joblib och Pillow
- Microsoft VS Code
- Git och GitHub
- Typora för redigering av Markdown
- Booli-data från separat projekt:
  `https://github.com/Kazzon-gbg/BostadsData_Partille`

GitHub-länk till projektet:

```text
https://github.com/Kazzon-gbg/Bostadspris-prediktion-i-Partille-kommun
```
