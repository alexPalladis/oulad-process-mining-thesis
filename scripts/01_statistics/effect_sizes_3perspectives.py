"""Cliff's delta for the three perspectives (Section 3.3), Pass vs Fail per course.
Forum clicks are computed for all students with >= 1 VLE click (non-users = 0).
Output: results/statistics/effect_sizes_3perspectives.csv"""
import os
import pandas as pd
from scipy.stats import rankdata

RAW = "../../data/raw"
OUT = "../../results/statistics"
os.makedirs(OUT, exist_ok=True)


def cliffs_delta(x, y):
    """Ίδιος τύπος με control_flow.py (μέσω rank-sum, χειρίζεται ισοπαλίες)."""
    x, y = list(x), list(y)
    nx, ny = len(x), len(y)
    ranks = rankdata(x + y)
    rx = sum(ranks[:nx])
    auc = (rx - nx * (nx + 1) / 2) / (nx * ny)
    return 2 * auc - 1


vle = pd.read_csv(f"{RAW}/vle.csv", usecols=["id_site", "code_module", "code_presentation", "activity_type"])
info = pd.read_csv(f"{RAW}/studentInfo.csv", usecols=["code_module", "code_presentation", "id_student", "final_result"])
svle = pd.read_csv(f"{RAW}/studentVle.csv",
                   usecols=["code_module", "code_presentation", "id_student", "id_site", "sum_click"])
assessments = pd.read_csv(f"{RAW}/assessments.csv")
studentAssessment = pd.read_csv(f"{RAW}/studentAssessment.csv")

courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()
rows = []

for module, pres in courses:
    vle_c = vle[(vle.code_module == module) & (vle.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)
                  & info.final_result.isin(["Pass", "Fail"])]
    svle_c = svle[(svle.code_module == module) & (svle.code_presentation == pres)]

    ev = svle_c.merge(vle_c[["id_site", "activity_type"]], on="id_site", how="left")
    ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")

    ev["forum_click"] = ev["sum_click"].where(ev.activity_type == "forumng", 0)
    ps = ev.groupby(["id_student", "final_result"])[["sum_click", "forum_click"]].sum().reset_index()
    ps["forum_share"] = ps["forum_click"] / ps["sum_click"]

    P, F = ps[ps.final_result == "Pass"], ps[ps.final_result == "Fail"]
    if len(P) < 5 or len(F) < 5:
        continue

    Pu, Fu = P[P.forum_click > 0], F[F.forum_click > 0]   # same population as organizational.py

    a_c = assessments[(assessments.code_module == module) & (assessments.code_presentation == pres)]
    sa = studentAssessment.merge(a_c[["id_assessment"]], on="id_assessment", how="inner")
    first = sa.groupby("id_student")["date_submitted"].min().reset_index()
    first = first.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")
    fP = first[first.final_result == "Pass"]["date_submitted"]
    fF = first[first.final_result == "Fail"]["date_submitted"]

    rows.append({
        "course": f"{module}-{pres}",
        "n_pass": len(P), "n_fail": len(F),
        "delta_total": round(cliffs_delta(P.sum_click, F.sum_click), 3),
        "n_pass_forum": len(Pu), "n_fail_forum": len(Fu),
        "ratio_forum_users": round(Pu.forum_click.mean() / Fu.forum_click.mean(), 2) if len(Fu) else None,
        "delta_forum_users": round(cliffs_delta(Pu.forum_click, Fu.forum_click), 3) if len(Pu) >= 5 and len(Fu) >= 5 else None,
        "delta_forum_all": round(cliffs_delta(P.forum_click, F.forum_click), 3),
        "delta_forum_share": round(cliffs_delta(P.forum_share, F.forum_share), 3),
        "pct_fail_no_forum": round(100 * (F.forum_click == 0).mean(), 1),
        "pct_pass_no_forum": round(100 * (P.forum_click == 0).mean(), 1),
        "delta_first_sub": round(cliffs_delta(fP, fF), 3) if len(fP) >= 5 and len(fF) >= 5 else None,
    })

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/effect_sizes_3perspectives.csv", index=False)

pd.set_option("display.width", 250)
print(res.to_string(index=False))

print("\n=== ΣΥΝΟΨΗ στα", len(res), "μαθήματα (min / mean / max) ===")
for col in ["delta_total", "delta_forum_users", "delta_forum_all", "delta_forum_share", "delta_first_sub"]:
    s = res[col].dropna()
    print(f"{col:20s} min={s.min():7.3f}  mean={s.mean():7.3f}  max={s.max():7.3f}  (n={len(s)})")
print("\nΜαθήματα όπου |δ forum_all| > δ total:",
      int((res.delta_forum_all.abs() > res.delta_total.abs()).sum()), "/", len(res))
print("Κατώφλια Romano: |δ|>=0.147 μικρό, >=0.33 μεσαίο, >=0.474 μεγάλο")