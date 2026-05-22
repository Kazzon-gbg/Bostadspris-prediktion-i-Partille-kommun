# Slutrapport: jämförelse mellan 1, 2 och 3 år

Skapad: 2026-05-22 11:08

## Syfte

Denna slutrapport jämför tre separata modellrapporter:

- Rapport 1: modell tränad på senaste 1 året
- Rapport 2: modell tränad på senaste 2 åren
- Rapport 3: modell tränad på hela datasetet, cirka 3 år

Syftet är att undersöka om bostadsprismodellen blir bättre av mer historisk
data eller av kortare och mer aktuell data.

## Sammanfattning

| Period | Bästa modell enligt RMSE | Rader | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|---:|
| 1 år | Random Forest log-target | 295 | 770 235 kr | 1 102 927 kr | 0.553 | 12.4 % |
| 2 år | Gradient Boosting log-target | 599 | 710 711 kr | 1 030 689 kr | 0.689 | 11.3 % |
| 3 år | Random Forest log-target | 948 | 615 788 kr | 968 072 kr | 0.732 | 10.3 % |

Resultatet visar att 3-årsmodellen presterar bäst i denna jämförelse. Den har
lägst MAE, lägst RMSE, högst R² och lägst MAPE.

## Modelljämförelser per period

### 1 år

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Random Forest log-target | 770 235 kr | 1 102 927 kr | 0.553 | 12.4 % |
| Linear Regression log-target | 777 273 kr | 1 114 692 kr | 0.543 | 11.7 % |
| Gradient Boosting log-target | 829 196 kr | 1 177 665 kr | 0.490 | 12.6 % |
| Decision Tree log-target | 964 996 kr | 1 258 845 kr | 0.417 | 15.8 % |
### 2 år

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Gradient Boosting log-target | 710 711 kr | 1 030 689 kr | 0.689 | 11.3 % |
| Random Forest log-target | 700 742 kr | 1 054 690 kr | 0.674 | 11.0 % |
| Linear Regression log-target | 816 522 kr | 1 137 669 kr | 0.621 | 13.1 % |
| Decision Tree log-target | 870 740 kr | 1 221 806 kr | 0.563 | 13.6 % |
### 3 år

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Random Forest log-target | 615 788 kr | 968 072 kr | 0.732 | 10.3 % |
| Gradient Boosting log-target | 619 088 kr | 975 959 kr | 0.727 | 10.3 % |
| Linear Regression log-target | 681 125 kr | 1 050 154 kr | 0.684 | 11.6 % |
| Decision Tree log-target | 737 498 kr | 1 120 247 kr | 0.641 | 12.0 % |

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
