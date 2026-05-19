# Bostadspris-prediktion i Partille kommun med Python och AI



Detta projekt är ett examensarbete för kursen **Python for AI**. Projektet
undersöker om det går att prediktera slutpris för bostäder i **Partille kommun**,
inklusive **Sävedalen**, med hjälp av verklig bostadsdata och maskininlärning.

Projektet använder en färdig CSV-fil med bostadsdata. Själva hämtningen av data
från Booli ingår **inte** i detta projekt, utan hanteras separat i ett annat
script/projekt.

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

Den huvudsakliga datafilen är:

```text
data/partille_housing_real_2023_today.csv
```

Datasetet består av verkliga bostadsförsäljningar i Partille kommun från 
**2023-01-02 till 2026-05-18**.

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
Helt tomma metadatafält filtreras också bort eftersom de inte tillför stabil
information.

## Avgränsning: datahämtning ingår inte

Det här projektet fokuserar på:

- dataförståelse
- datarensning inför modellering
- feature engineering
- visualisering
- modellträning
- modellutvärdering
- rapportering

Själva insamlingen av data från Booli ingår inte i projektet. Den hanteras
separat och resultatet från den processen är CSV-filen:

```text
data/partille_housing_real_2023_today.csv
```

Det gör projektet tydligare som ett maskininlärningsprojekt: modellen tränas på
en färdig dataset, medan datahämtning är ett separat förarbete.

## Projektstruktur

```text
.
├── data/
│   └── partille_housing_real_2023_today.csv
├── src/
│   └── train_model_partille.py
├── reports/
│   └── resultat.md
├── outputs/
│   ├── figures/
│   └── models/
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

Om `requirements.txt` inte är uppdaterad kan följande paket installeras manuellt:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Kör projektet

Kör modellträningen:

```bash
python src/train_model_partille.py
```

Scriptet gör följande:

1. Läser in CSV-filen.
2. Gör grundläggande datakontroll.
3. Skapar extra features.
4. Skapar visualiseringar.
5. Delar upp datan i träningsdata och testdata.
6. Tränar flera regressionsmodeller.
7. Utvärderar modellerna.
8. Sparar bästa modellen.
9. Skapar en resultatrapport.

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

MAPE är särskilt pedagogiskt för bostadspriser eftersom det visar felet i
procent i stället för bara kronor.

## Modellresultat

I den förbättrade modellversionen med feature engineering, tydligare feature-urval
och log-transformering
uppnåddes ungefär följande nivå:

```text
Bästa modell: Random Forest log-target
MAE: cirka 616 000 kr
RMSE: cirka 968 000 kr
R²: cirka 0.73
MAPE: cirka 10 %
```

Resultatet kan variera något beroende på exakt dataversion och train/test-split.

## Förbättringsförslag

Nästa steg för att förbättra modellen är främst att förstärka datan snarare än
att bara byta algoritm:

- komplettera med skick, renoveringsår, energiklass och driftskostnad där det går
- lägga till mer exakt lägesinformation, exempelvis avstånd till skola, service och kollektivtrafik
- validera modellen med cross-validation så resultatet blir mindre beroende av en enda train/test-split
- jämföra resultat med och utan `asking_price_sek`, eftersom utgångspris ofta är en stark men ibland saknad feature
- undersöka feature importance för bästa modellen och ta bort features som mest skapar brus

## Figurer

Scriptet sparar figurer i:

```text
outputs/figures/
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

## Resultatrapport

Efter körning skapas rapporten:

```text
reports/resultat.md
```

Rapporten innehåller:

- sammanfattning
- syfte
- datasetbeskrivning
- hantering av saknade värden
- val av X och y
- feature engineering
- modelljämförelse
- resultat
- tolkning
- begränsningar
- etisk reflektion

## Begränsningar

Även om projektet använder verklig data från mer än tre år är bostadspriser
svåra att prediktera exakt. Viktiga faktorer saknas fortfarande, till exempel:

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

En bostadsprismodell kan påverka ekonomiska beslut. Därför bör resultatet
användas med försiktighet. Om datan är ofullständig eller inte representerar
alla områden och bostadstyper rättvist kan modellen ge missvisande resultat.

Modellen bör inte användas som ensam grund för köp, försäljning eller
lånebeslut. En mänsklig bedömning bör alltid finnas kvar.

## Kort slutsats

Projektet visar ett komplett arbetsflöde för ett AI-/maskininlärningsprojekt i
Python:

- dataimport från CSV
- dataförståelse
- feature engineering
- visualisering
- preprocessing
- modellträning
- utvärdering
- rapportering
- etisk reflektion

Resultatet visar att det går att skapa en användbar modell för bostadspriser i
Partille kommun, men också att modellens kvalitet beror mer på relevanta
features än på antalet kolumner.
