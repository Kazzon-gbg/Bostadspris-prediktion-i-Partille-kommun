# Struktur for inlamningsuppgiften

Detta dokument kopplar kursens inlamningsuppgift till projektet
`Bostadspris-prediktion-i-Partille-kommun`.

## Kort sammanfattning

Projektet uppfyller uppgiftens huvudkrav genom att:

- valja ett tydligt regressionsproblem: prediktera slutpris for bostader
- anvanda ett eget verkligt dataset med bostadsforsaljningar i Partille kommun
- forbereda data med rensning, imputering, encoding och feature engineering
- trana och jamfora flera modeller med Scikit-learn
- utvardera modellerna med MAE, RMSE, R2 och MAPE
- skapa figurer, Markdown-rapporter och PDF-rapporter
- reflektera over begransningar, bias och verklig anvandning

## Rekommenderad inlamning

Lamna in:

1. GitHub-lank till repositoryt:

   ```text
   https://github.com/Kazzon-gbg/Bostadspris-prediktion-i-Partille-kommun
   ```

2. PDF-rapport:

   ```text
   reports/report_3years_260522_1055.pdf
   ```

3. Som extra underlag kan du aven namna jamforelserapporten:

   ```text
   reports/report_jamforelse_260522_1104.pdf
   ```

Huvudrapporten bor vara 3-arsrapporten eftersom den ar projektets basta modell
och innehaller hela arbetsflodet fran problem och dataset till resultat och
etisk reflektion.

## Koppling till uppgiftens steg

| Uppgiftskrav | Hur projektet uppfyller kravet | Viktiga filer |
|---|---|---|
| Val av problem | Prediktera slutpris for bostader i Partille kommun | `README.md`, `reports/report_3years_260521_2348.md` |
| Val av dataset | Verklig bostadsdata fran Partille, 2023-01-02 till 2026-05-18 | `data/partille_housing_real_2023_today.csv` |
| Motivering | Datasetet passar eftersom det innehaller bostadsegenskaper och slutpris | `README.md`, rapportens avsnitt 3-4 |
| Klassificering eller regression | Regression, eftersom target ar ett numeriskt slutpris | `README.md`, rapportens avsnitt 3 |
| Ladda in data | CSV lases in med Pandas | `src/modules/functions.py` |
| Hantera saknade varden | Medianimputering for numeriska varden och vanligaste varde for kategoriska | `src/modules/functions.py` |
| Kategoriska variabler | One-hot encoding med `OneHotEncoder` | `src/modules/functions.py` |
| Feature-val | Lackage- och metadatakolumner exkluderas | `src/modules/config.py`, `src/modules/functions.py` |
| Visualisering | Prisfordelning, boarea mot pris, korrelation, residualer med mera | `outputs/figures/` |
| Train/test split | 80/20-split med `train_test_split` | `src/modules/project.py` |
| Modelltraning | Linear Regression, Decision Tree, Random Forest och Gradient Boosting | `src/modules/functions.py` |
| Utvardering | MAE, RMSE, R2 och MAPE | `src/modules/functions.py`, rapportens avsnitt 9 |
| Reflektion | Begransningar, felkallor och etisk anvandning | `README.md`, rapportens avsnitt 11-12 |

## Rapportstruktur enligt uppgiften

Om du vill korta ner rapporten till cirka 1-2 sidor kan du anvanda denna
disposition.

### 1. Problem och dataset

Jag har valt att undersoka om det gar att prediktera slutpris for bostader i
Partille kommun med hjalp av maskininlarning. Datasetet bestar av verkliga
bostadsforsaljningar fran perioden 2023-01-02 till 2026-05-18. Eftersom
slutpriset ar ett numeriskt varde ar detta ett regressionsproblem.

### 2. Dataforberedelse

Datan innehaller saknade varden, exempelvis for utgangspris, antal rum, biarea
och tomtarea. Numeriska saknade varden hanteras med medianimputering och
kategoriska varden med vanligaste varde. Kategoriska variabler omvandlas med
one-hot encoding. Kolumner som kan ge datalackage, till exempel
`price_per_m2` och prisforandringskolumner, tas bort fran modellens features.

### 3. Modell

Projektet testar flera regressionsmodeller i Scikit-learn: Linear Regression,
Decision Tree, Random Forest och Gradient Boosting. Slutpriset log-transformeras
under traningen for att minska effekten av mycket dyra objekt. Den basta modellen
for huvudkornigen blev Random Forest med log-transformerad target.

### 4. Resultat

3-arsmodellen presterade bast:

| Matt | Resultat |
|---|---:|
| MAE | 615 788 kr |
| RMSE | 968 072 kr |
| R2 | 0.732 |
| MAPE | 10.3 % |

Det betyder att modellen forklarar en relativt stor del av variationen i
slutpris, men att felet fortfarande kan vara omkring en miljon kronor for vissa
objekt. MAPE pa 10.3 procent gor resultatet enklare att tolka eftersom bostader
ligger pa olika prisnivaer.

### 5. Begransningar och verklig anvandning

Modellen saknar viktiga faktorer som skick, renoveringsstandard, exakt mikrolage,
planlosning, utsikt, narhet till service och aktuellt rantelage. Den bor darfor
inte anvandas som ett facit for kop, forsaljning eller vardering. Den kan
anvandas som ett analytiskt stod tillsammans med mansklig bedomning.

## G-kriterier

Projektet uppfyller kraven for Godkant:

- tydligt problem och dataset
- fungerande modell
- hanterad och anpassad data
- relevant utvardering
- rapport med process och resultat
- kodfiler som visar hela arbetsflodet

## VG-starkande delar

Projektet har flera delar som starker det mot VG:

- flera alternativa modeller jamfors
- feature engineering anvands
- log-transformerad target anvands
- resultat jamfors mellan 1, 2 och 3 ars data
- rapporten diskuterar begransningar och forbattrningar
- koden ar uppdelad i moduler och en samordnande klass
- bade Markdown- och PDF-rapporter skapas automatiskt

Det som skulle kunna starka projektet ytterligare ar cross-validation och en
kortare feature-importance-analys for basta modellen.

## Checklista fore inlamning

- [ ] Kontrollera att GitHub-repot ar publikt, eller ge lararen atkomst om det ar privat.
- [ ] Kontrollera att `README.md` beskriver hur projektet kors.
- [ ] Ladda upp `reports/report_3years_260522_1055.pdf` i ItsLearning.
- [ ] Klistra in GitHub-lanken i textfaltet i ItsLearning.
- [ ] Namn eventuellt `reports/report_jamforelse_260522_1104.pdf` som extra underlag.
- [ ] Skriv kort i inlamningsrutan att projektet ar ett regressionsprojekt for bostadspriser.

## Forslag pa text i ItsLearning

```text
Jag lamnar in mitt projekt "Bostadspris-prediktion i Partille kommun".

Projektet anvander verklig bostadsdata fran Partille kommun och bygger en
regressionsmodell som predikterar slutpris. I repositoryt finns kod, dataset,
figurer, modeller, README och rapporter. Jag laddar upp huvudrapporten som PDF
och bifogar GitHub-lanken har:

https://github.com/Kazzon-gbg/Bostadspris-prediktion-i-Partille-kommun
```
