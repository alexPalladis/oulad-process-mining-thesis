import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
import os

RAW = "../../data/raw"
OUT = "../../results/statistics"
os.makedirs(OUT, exist_ok=True)

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    ranks = rankdata(list(x) + list(y))
    rx = sum(ranks[:nx])
    auc = (rx - nx*(nx+1)/2) / (nx*ny)
    return 2*auc - 1

vle = pd.read_csv(f"{RAW}/vle.csv")
info = pd.read_csv(f"{RAW}/studentInfo.csv")
svle = pd.read_csv(f"{RAW}/studentVle.csv")

courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()

PAIRS = [("Distinction", "Fail"), ("Withdrawn", "Fail"), ("Distinction", "Withdrawn")]

results = []
for module, pres in courses:
    vle_c = vle[(vle.code_module == module) & (vle.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]
    svle_c = svle[(svle.code_module == module) & (svle.code_presentation == pres)]

    ev = svle_c.merge(vle_c[["id_site", "activity_type"]], on="id_site", how="left")
    ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")
    per_student = ev.groupby(["id_student", "final_result"])["sum_click"].sum().reset_index()

    for g1, g2 in PAIRS:
        v1 = per_student[per_student.final_result == g1]["sum_click"]
        v2 = per_student[per_student.final_result == g2]["sum_click"]
        if len(v1) < 5 or len(v2) < 5:
            continue
        _, p = mannwhitneyu(v1, v2)
        delta = cliffs_delta(v1, v2)
        results.append({
            "course": f"{module}-{pres}", "pair": f"{g1} vs {g2}",
            "n1": len(v1), "n2": len(v2),
            "mean1": round(v1.mean(), 1), "mean2": round(v2.mean(), 1),
            "p_value": p, "cliffs_delta": round(delta, 3),
        })

summary = pd.DataFrame(results)
summary.to_csv(f"{OUT}/control_flow_pairwise_summary.csv", index=False)
print(summary.groupby("pair")["cliffs_delta"].describe())