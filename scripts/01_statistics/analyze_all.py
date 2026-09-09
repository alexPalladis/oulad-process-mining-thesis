import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    ranks = rankdata(list(x) + list(y))
    rx = sum(ranks[:nx])
    auc = (rx - nx*(nx+1)/2) / (nx*ny)
    return 2*auc - 1

vle = pd.read_csv("vle.csv")
info = pd.read_csv("studentInfo.csv")
svle = pd.read_csv("studentVle.csv")

courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()

results = []
for module, pres in courses:
    vle_c = vle[(vle.code_module == module) & (vle.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]
    svle_c = svle[(svle.code_module == module) & (svle.code_presentation == pres)]

    ev = svle_c.merge(vle_c[["id_site", "activity_type"]], on="id_site", how="left")
    ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")
    ev_pf = ev[ev.final_result.isin(["Pass", "Fail"])]

    per_student = ev_pf.groupby(["id_student", "final_result"])["sum_click"].sum().reset_index()
    fail = per_student[per_student.final_result == "Fail"]["sum_click"]
    pas = per_student[per_student.final_result == "Pass"]["sum_click"]

    if len(fail) < 5 or len(pas) < 5:
        continue  # πολύ μικρό δείγμα, παράλειψέ το

    _, p = mannwhitneyu(fail, pas)
    delta = cliffs_delta(pas, fail)

    results.append({
        "course": f"{module}-{pres}",
        "n_fail": len(fail), "n_pass": len(pas),
        "avg_clicks_fail": round(fail.mean(), 1),
        "avg_clicks_pass": round(pas.mean(), 1),
        "ratio": round(pas.mean() / fail.mean(), 2),
        "p_value": p,
        "cliffs_delta": round(delta, 3),
    })

summary = pd.DataFrame(results)
summary.to_csv("all_courses_summary.csv", index=False)
print(summary.to_string(index=False))