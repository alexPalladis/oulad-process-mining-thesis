import pandas as pd
from scipy.stats import mannwhitneyu

assessments = pd.read_csv("assessments.csv")
studentAssessment = pd.read_csv("studentAssessment.csv")
info = pd.read_csv("studentInfo.csv")

results = []
courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()

for module, pres in courses:
    a_c = assessments[(assessments.code_module == module) & (assessments.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]

    # Πρώτη αξιολόγηση κάθε φοιτητή (μικρότερο date_submitted)
    sa = studentAssessment.merge(a_c[["id_assessment"]], on="id_assessment", how="inner")
    first_sub = sa.groupby("id_student")["date_submitted"].min().reset_index()
    first_sub = first_sub.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")
    first_sub_pf = first_sub[first_sub.final_result.isin(["Pass", "Fail"])]

    fail = first_sub_pf[first_sub_pf.final_result == "Fail"]["date_submitted"]
    pas = first_sub_pf[first_sub_pf.final_result == "Pass"]["date_submitted"]

    if len(fail) < 5 or len(pas) < 5:
        continue

    _, p = mannwhitneyu(fail, pas)
    results.append({
        "course": f"{module}-{pres}",
        "avg_first_submission_day_fail": round(fail.mean(), 1),
        "avg_first_submission_day_pass": round(pas.mean(), 1),
        "p_value": p,
    })

summary = pd.DataFrame(results)
summary.to_csv("performance_summary.csv", index=False)
print(summary.to_string(index=False))