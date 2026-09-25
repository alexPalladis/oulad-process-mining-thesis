import pandas as pd
from scipy.stats import kruskal
import os

RAW = "../../data/raw"
OUT = "../../results/statistics"
os.makedirs(OUT, exist_ok=True)

vle = pd.read_csv(f"{RAW}/vle.csv")
info = pd.read_csv(f"{RAW}/studentInfo.csv")
svle = pd.read_csv(f"{RAW}/studentVle.csv")

courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()
GROUPS = ["Distinction", "Pass", "Withdrawn", "Fail"]

results = []
for module, pres in courses:
    vle_c = vle[(vle.code_module == module) & (vle.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]
    svle_c = svle[(svle.code_module == module) & (svle.code_presentation == pres)]

    ev = svle_c.merge(vle_c[["id_site", "activity_type"]], on="id_site", how="left")
    ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")

    per_student = ev.groupby(["id_student", "final_result"])["sum_click"].sum().reset_index()

    groups_data = {}
    for g in GROUPS:
        vals = per_student[per_student.final_result == g]["sum_click"]
        groups_data[g] = vals
        if len(vals) >= 5:
            results.append({
                "course": f"{module}-{pres}", "group": g,
                "n": len(vals), "mean_clicks": round(vals.mean(), 1),
                "median_clicks": round(vals.median(), 1),
            })

    valid = [groups_data[g] for g in GROUPS if len(groups_data[g]) >= 5]
    if len(valid) == 4:
        stat, p = kruskal(*valid)
        results.append({"course": f"{module}-{pres}", "group": "KRUSKAL_WALLIS",
                         "n": None, "mean_clicks": None, "median_clicks": p})

summary = pd.DataFrame(results)
summary.to_csv(f"{OUT}/control_flow_4groups_summary.csv", index=False)
print(summary.to_string(index=False))