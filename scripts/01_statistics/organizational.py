import pandas as pd
from scipy.stats import mannwhitneyu
import os

RAW = "../../data/raw"
OUT = "../../results/statistics"
os.makedirs(OUT, exist_ok=True)

vle = pd.read_csv(f"{RAW}/vle.csv")
info = pd.read_csv(f"{RAW}/studentInfo.csv")
svle = pd.read_csv(f"{RAW}/studentVle.csv")

results = []
courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()

for module, pres in courses:
    vle_c = vle[(vle.code_module == module) & (vle.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]
    svle_c = svle[(svle.code_module == module) & (svle.code_presentation == pres)]

    forum_sites = vle_c[vle_c.activity_type == "forumng"]["id_site"]
    ev = svle_c[svle_c.id_site.isin(forum_sites)]
    ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")
    ev_pf = ev[ev.final_result.isin(["Pass", "Fail"])]

    per_student = ev_pf.groupby(["id_student", "final_result"])["sum_click"].sum().reset_index()
    fail = per_student[per_student.final_result == "Fail"]["sum_click"]
    pas = per_student[per_student.final_result == "Pass"]["sum_click"]

    if len(fail) < 5 or len(pas) < 5:
        continue

    _, p = mannwhitneyu(fail, pas)
    results.append({
        "course": f"{module}-{pres}",
        "avg_forum_clicks_fail": round(fail.mean(), 1),
        "avg_forum_clicks_pass": round(pas.mean(), 1),
        "p_value": p,
    })

summary = pd.DataFrame(results)
summary.to_csv(f"{OUT}/organizational_summary.csv", index=False)
print(summary.to_string(index=False))