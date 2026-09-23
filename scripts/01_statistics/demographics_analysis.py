import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

RAW = "../../data/raw"
OUT = "../../results/statistics"
os.makedirs(OUT, exist_ok=True)

info = pd.read_csv(f"{RAW}/studentInfo.csv")

DEMO_COLS = ["gender", "age_band", "disability", "imd_band", "highest_education", "region"]

def cramers_v(chi2, n, table_shape):
    """Cramér's V effect size για πίνακα συνάφειας r x c."""
    r, c = table_shape
    min_dim = min(r - 1, c - 1)
    return np.sqrt(chi2 / (n * min_dim))

print("Συνολικοί φοιτητές:", len(info))
print("Κατανομή τελικής έκβασης:")
print(info.final_result.value_counts())
print()

results = []
for col in DEMO_COLS:
    # Πίνακας συνάφειας: δημογραφική κατηγορία x τελική έκβαση
    ct = pd.crosstab(info[col], info["final_result"])
    chi2, p, dof, expected = chi2_contingency(ct)

    n = ct.values.sum()
    v = cramers_v(chi2, n, ct.shape)

    # Ποσοστό (%) ανά κατηγορία-έκβαση, για ερμηνεία
    pct = pd.crosstab(info[col], info["final_result"], normalize="index") * 100
    pct = pct.round(1)

    print(f"=== {col} ===")
    print("Αριθμός κατηγοριών:", ct.shape[0], " chi2 =", round(chi2, 1),
          " p =", p, " Cramér's V =", round(v, 3))
    print(pct)
    print()

    results.append({
        "variable": col, "n_categories": ct.shape[0],
        "chi2": chi2, "p_value": p, "dof": dof,
        "cramers_v": round(v, 3),
    })

summary = pd.DataFrame(results)
summary.to_csv(f"{OUT}/demographics_summary.csv", index=False)

# Αναλυτικά ποσοστά ανά μεταβλητή, σε ξεχωριστό αρχείο
all_pct = []
for col in DEMO_COLS:
    pct = pd.crosstab(info[col], info["final_result"], normalize="index") * 100
    pct = pct.round(1).reset_index().rename(columns={col: "category"})
    pct.insert(0, "variable", col)
    all_pct.append(pct)
pd.concat(all_pct, ignore_index=True).to_csv(f"{OUT}/demographics_percentages.csv", index=False)

print(f"Έτοιμο. Αρχεία στο {OUT}/: demographics_summary.csv (με Cramér's V), demographics_percentages.csv")