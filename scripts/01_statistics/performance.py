import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import os

RAW = "../../data/raw"
OUT = "../../results/statistics"
os.makedirs(OUT, exist_ok=True)

assessments = pd.read_csv(f"{RAW}/assessments.csv")
studentAssessment = pd.read_csv(f"{RAW}/studentAssessment.csv")
studentAssessment = studentAssessment[studentAssessment.is_banked == 0]  # banked = carried over from a previous presentation
info = pd.read_csv(f"{RAW}/studentInfo.csv")

results = []
courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()

for module, pres in courses:
    a_c = assessments[(assessments.code_module == module) & (assessments.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]
    info_pf = info_c[info_c.final_result.isin(["Pass", "Fail"])]

    sa = studentAssessment.merge(a_c[["id_assessment"]], on="id_assessment", how="inner")
    first_sub = sa.groupby("id_student")["date_submitted"].min().reset_index()
    first_sub_pf = first_sub.merge(info_pf[["id_student", "final_result"]], on="id_student", how="inner")

    fail = first_sub_pf[first_sub_pf.final_result == "Fail"]["date_submitted"]
    pas = first_sub_pf[first_sub_pf.final_result == "Pass"]["date_submitted"]

    if len(fail) < 5 or len(pas) < 5:
        continue

    _, p = mannwhitneyu(fail, pas)
    results.append({
        "course": f"{module}-{pres}",
        "n_fail": len(fail), "n_pass": len(pas),
        "n_fail_no_submission": int((info_pf.final_result == "Fail").sum() - len(fail)),
        "n_pass_no_submission": int((info_pf.final_result == "Pass").sum() - len(pas)),
        "avg_first_submission_day_fail": round(fail.mean(), 1),
        "avg_first_submission_day_pass": round(pas.mean(), 1),
        "diff_days": round(fail.mean() - pas.mean(), 2),
        "p_value": p,
    })

summary = pd.DataFrame(results)

n = len(summary)
order = np.argsort(summary.p_value.values)
adj = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * summary.p_value.values[order]))
summary["p_holm"] = np.empty(n)
summary.loc[summary.index[order], "p_holm"] = adj

summary.to_csv(f"{OUT}/performance_summary.csv", index=False)
pd.set_option("display.width", 250)
print(summary.to_string(index=False))

print("\n=== SUMMARY ===")
print(f"mean difference (Fail - Pass): {summary.diff_days.mean():.2f} days "
      f"(range {summary.diff_days.min():.2f} to {summary.diff_days.max():.2f})")
print(f"courses p<0.001: {(summary.p_value < 0.001).sum()}/{n} | p<0.05: {(summary.p_value < 0.05).sum()}/{n} "
      f"| Holm p<0.05: {(summary.p_holm < 0.05).sum()}/{n}")
print(f"courses where Fail submitted earlier (negative difference): {(summary.diff_days < 0).sum()}")
nf = info[info.final_result == "Fail"].shape[0]; npass = info[info.final_result == "Pass"].shape[0]
print(f"never submitted: Fail {summary.n_fail_no_submission.sum()} of {nf} "
      f"({100 * summary.n_fail_no_submission.sum() / nf:.1f}%), Pass {summary.n_pass_no_submission.sum()} of {npass}")