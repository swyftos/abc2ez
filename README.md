# README.md

# Test de changement de régime dans la politique monétaire US (1913–2023)

Ce projet propose un script Python permettant de tester si la relation entre le **déficit public/PIB** et le **taux directeur de la Federal Reserve** change après l’**Accord Treasury–Federal Reserve de 1951**.

---

## 📂 Contenu du dépôt

- `test_regime_fed_deficit_1951.py` : script principal Python.
- `Fed_Taux_Dette_Deficit_1913_2023.xlsx` : fichier Excel avec les données (taux Fed, dette/PIB, déficit/PIB).
- Résultats générés par le script :
  - `resume_OLS.txt` : résumé complet de la régression OLS.
  - `resume_HAC.txt` : résumé avec erreurs robustes Newey–West (HAC).
  - `resultats_regression_interaction_post1951.csv` : coefficients OLS et HAC.
  - `test_chow_1951.txt` : résultats du test de Chow (rupture structurelle en 1951).
  - `scatter_deficit_vs_taux.png` : nuage de points déficit/PIB vs taux directeur (avant/après 1951).
  - `serie_taux_deficit.png` : séries temporelles taux Fed & déficit/PIB.

---

## ⚙️ Installation

### Sous Linux / MacOS

```bash

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
