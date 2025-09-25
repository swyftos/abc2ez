#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de changement de régime (post-1951) dans la relation Déficit/PIB → Taux directeur de la Fed

Usage (terminal) :
    python test_regime_fed_deficit_1951.py --excel Fed_Taux_Dette_Deficit_1913_2023.xlsx --breakyear 1951

Sorties générées :
    - resume_OLS.txt / resume_HAC.txt
    - resultats_regression_interaction_post1951.csv
    - test_chow_<breakyear>.txt
    - scatter_deficit_vs_taux.png
    - serie_taux_deficit.png
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import f as fdist

EXPECTED_COLS = ["Année", "Taux d'intérêt de la Fed (%)", "Dette/PIB (%)", "Déficit/PIB (%)"]

# ---------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------
def load_data(path_excel: str) -> pd.DataFrame:
    if not os.path.exists(path_excel):
        print(f"[ERREUR] Fichier introuvable : {path_excel}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_excel(path_excel)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes : {missing}\nColonnes trouvées : {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    return df.sort_values("Année").reset_index(drop=True)

# ---------------------------------------------------------------------
# Modèle avec interaction post-1951
# ---------------------------------------------------------------------
def fit_interaction_model(df: pd.DataFrame, breakyear: int):
    df = df.copy()
    df["post"] = (df["Année"] > breakyear).astype(int)
    df["interaction"] = df["Déficit/PIB (%)"] * df["post"]

    Y = df["Taux d'intérêt de la Fed (%)"]
    X = sm.add_constant(df[["Déficit/PIB (%)", "post", "interaction"]])
    model = sm.OLS(Y, X).fit()

    # Erreurs robustes HAC (Newey–West), 2 lags
    model_hac = model.get_robustcov_results(cov_type="HAC", maxlags=2)
    return df, model, model_hac

# ---------------------------------------------------------------------
# Test de Chow (modèle simple : i_t ~ const + déficit_t)
# ---------------------------------------------------------------------
def chow_test_simple(df: pd.DataFrame, breakyear: int):
    pre = df["Année"] <= breakyear
    post = df["Année"] > breakyear

    Y_pre = df.loc[pre, "Taux d'intérêt de la Fed (%)"]
    X_pre = sm.add_constant(df.loc[pre, ["Déficit/PIB (%)"]])

    Y_post = df.loc[post, "Taux d'intérêt de la Fed (%)"]
    X_post = sm.add_constant(df.loc[post, ["Déficit/PIB (%)"]])

    Y_all = df["Taux d'intérêt de la Fed (%)"]
    X_all = sm.add_constant(df[["Déficit/PIB (%)"]])

    mod_pre = sm.OLS(Y_pre, X_pre).fit()
    mod_post = sm.OLS(Y_post, X_post).fit()
    mod_all = sm.OLS(Y_all, X_all).fit()

    SSE_pre = float(np.sum(mod_pre.resid**2))
    SSE_post = float(np.sum(mod_post.resid**2))
    SSE_all = float(np.sum(mod_all.resid**2))

    N1, N2 = len(Y_pre), len(Y_post)
    k = X_pre.shape[1]  # const + 1 régresseur = 2

    num = (SSE_all - (SSE_pre + SSE_post)) / k
    den = (SSE_pre + SSE_post) / (N1 + N2 - 2 * k)
    F_stat = num / den
    p_val = 1 - fdist.cdf(F_stat, dfn=k, dfd=(N1 + N2 - 2 * k))
    return F_stat, p_val

# ---------------------------------------------------------------------
# Export des résultats & graphiques
# ---------------------------------------------------------------------
def export_results(model, model_hac, F_stat, p_val, breakyear: int):
    # Résumés
    with open("resume_OLS.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    with open("resume_HAC.txt", "w", encoding="utf-8") as f:
        f.write(model_hac.summary().as_text())

    with open(f"test_chow_{breakyear}.txt", "w", encoding="utf-8") as f:
        f.write(f"Test de Chow (rupture en {breakyear})\n")
        f.write(f"F = {F_stat:.4f}\n")
        f.write(f"p-value = {p_val:.6f}\n")
        f.write("Règle : rejeter H0 (pas de rupture) si p < 0.05.\n")

    # Tableau coefficients OLS & HAC (pas de .values sur les ndarrays HAC)
    out = {
        "variable": list(model.params.index),
        "coef_OLS": model.params.tolist(),
        "std_err_OLS": model.bse.tolist(),
        "t_OLS": model.tvalues.tolist(),
        "pval_OLS": model.pvalues.tolist(),
        "coef_HAC": model_hac.params,      # ndarray
        "std_err_HAC": model_hac.bse,      # ndarray
        "t_HAC": model_hac.tvalues,        # ndarray
        "pval_HAC": model_hac.pvalues,     # ndarray
    }
    # Convertit en DataFrame proprement
    df_out = pd.DataFrame(out)
    df_out.to_csv("resultats_regression_interaction_post1951.csv", index=False, encoding="utf-8")

def make_plots(df: pd.DataFrame, breakyear: int, model):
    # Diagnostics rapides
    dw = durbin_watson(model.resid)
    bp = het_breuschpagan(model.resid, model.model.exog)  # LM stat, LM p, F stat, F p
    print(f"Durbin–Watson : {dw:.2f} (≈2 attendu si pas d'autocorrélation)")
    print("Breusch–Pagan :", dict(zip(["LM stat","LM p","F stat","F p"], bp)))

    # Scatter déficit vs taux par période
    pre_mask = df["post"] == 0
    post_mask = df["post"] == 1

    plt.figure(figsize=(8, 5))
    plt.scatter(df.loc[pre_mask, "Déficit/PIB (%)"], df.loc[pre_mask, "Taux d'intérêt de la Fed (%)"],
                label=f"≤ {breakyear}", alpha=0.7)
    plt.scatter(df.loc[post_mask, "Déficit/PIB (%)"], df.loc[post_mask, "Taux d'intérêt de la Fed (%)"],
                label=f"> {breakyear}", alpha=0.7)
    plt.axvline(0, ls="--", lw=1, color="grey")
    plt.title("Taux de la Fed vs Déficit/PIB (nuage de points)")
    plt.xlabel("Déficit/PIB (%)")
    plt.ylabel("Taux d'intérêt de la Fed (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatter_deficit_vs_taux.png", dpi=180)
    plt.close()

    # Séries temporelles
    plt.figure(figsize=(10, 5))
    plt.plot(df["Année"], df["Taux d'intérêt de la Fed (%)"], label="Taux de la Fed (%)")
    plt.plot(df["Année"], df["Déficit/PIB (%)"], label="Déficit/PIB (%)")
    plt.axvline(breakyear, ls="--", lw=1, color="grey", label=f"Accord {breakyear}")
    plt.title("Séries temporelles : taux de la Fed & déficit/PIB")
    plt.xlabel("Année")
    plt.ylabel("Pourcentage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("serie_taux_deficit.png", dpi=180)
    plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test de rupture post-1951 dans la relation déficit/PIB → taux Fed.")
    parser.add_argument("--excel", type=str, required=True,
                        help="Chemin du fichier Excel (ex: Fed_Taux_Dette_Deficit_1913_2023.xlsx)")
    parser.add_argument("--breakyear", type=int, default=1951,
                        help="Année de rupture (par défaut: 1951).")
    args = parser.parse_args()

    df = load_data(args.excel)
    df2, model, model_hac = fit_interaction_model(df, args.breakyear)

    print(model.summary())
    print("\n=== Résumé HAC (Newey–West, maxlags=2) ===")
    print(model_hac.summary())

    F_stat, p_val = chow_test_simple(df, args.breakyear)
    print(f"\nTest de Chow (rupture en {args.breakyear}) : F = {F_stat:.3f} | p-value = {p_val:.6f}")

    export_results(model, model_hac, F_stat, p_val, args.breakyear)
    make_plots(df2, args.breakyear, model)

    print("\nFichiers générés :")
    for name in [
        "resume_OLS.txt",
        "resume_HAC.txt",
        "resultats_regression_interaction_post1951.csv",
        f"test_chow_{args.breakyear}.txt",
        "scatter_deficit_vs_taux.png",
        "serie_taux_deficit.png",
    ]:
        print(" -", name)

if __name__ == "__main__":
    main()
