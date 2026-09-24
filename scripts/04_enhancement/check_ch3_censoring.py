"""
check_ch3_censoring.py  --  robustness check for Section 3.4 (Withdrawn vs Fail)

Question: Withdrawn students are enrolled for fewer weeks than Fail students,
so their lower TOTAL clicks (Section 3.4) could partly reflect shorter
exposure rather than lower engagement. This script repeats the Withdrawn vs
Fail comparison in all 22 course presentations with two measures:

  (a) total_clicks      all clicks of the student in studentVle.csv
                        (intended to replicate the Section 3.4 measure)
  (b) clicks_per_week   clicks from day 0 to the end of the student's
                        enrolment, divided by the number of enrolled weeks.
                        End of enrolment = date_unregistration for Withdrawn,
                        module_presentation_length (courses.csv) otherwise.

Students without any VLE click are excluded (as in Section 3.4, to be
confirmed against control_flow_pairwise.py). Withdrawn students who
unregistered on or before day 0 have no enrolled time in the course and are
excluded from (b); their number is reported.

Tests: two-sided Mann-Whitney U and Cliff's delta (positive = Withdrawn higher).
Output: results/statistics/ch3_censoring_check.csv
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = None      # e.g. Path(r"C:\Users\Alex\Desktop\thesis_oulad\data\raw")
OUT_FILE = REPO_ROOT / "results" / "statistics" / "ch3_censoring_check.csv"
CHUNK_SIZE = 2_000_000

if RAW_DIR is None:
    hits = [p.parent for p in REPO_ROOT.rglob("studentInfo.csv") if "results" not in p.parts]
    if not hits:
        sys.exit(f"studentInfo.csv not found under {REPO_ROOT}. Set RAW_DIR.")
    RAW_DIR = hits[0]
RAW_DIR = Path(RAW_DIR)
os.makedirs(OUT_FILE.parent, exist_ok=True)

info = pd.read_csv(RAW_DIR / "studentInfo.csv")
info = info[info.final_result.isin(["Withdrawn", "Fail"])]
reg = pd.read_csv(RAW_DIR / "studentRegistration.csv", na_values=["?", ""])
courses = pd.read_csv(RAW_DIR / "courses.csv")
keys = ["code_module", "code_presentation", "id_student"]

# per-student totals, streamed from studentVle.csv
tot, enr = [], []
st = info.merge(reg, on=keys, how="left").merge(
    courses, on=["code_module", "code_presentation"])
st["end"] = np.where(st.final_result == "Withdrawn",
                     st.date_unregistration, st.module_presentation_length)
end_map = st.set_index(keys).end
for ch in pd.read_csv(RAW_DIR / "studentVle.csv", chunksize=CHUNK_SIZE):
    ch = ch.merge(st[keys + ["end"]], on=keys)
    if ch.empty:
        continue
    tot.append(ch.groupby(keys).sum_click.sum())
    w = ch[(ch.date >= 0) & (ch.date <= ch.end)]
    enr.append(w.groupby(keys).sum_click.sum())

total = pd.concat(tot).groupby(level=[0, 1, 2]).sum().rename("total_clicks")
in_window = pd.concat(enr).groupby(level=[0, 1, 2]).sum().rename("clicks_in_enrolment")
df = st.set_index(keys).join(total, how="inner").join(in_window)
df["clicks_in_enrolment"] = df.clicks_in_enrolment.fillna(0)
df["enrolled_weeks"] = df.end / 7
df["clicks_per_week"] = np.where(df.enrolled_weeks > 0,
                                 df.clicks_in_enrolment / df.enrolled_weeks, np.nan)
df = df.reset_index()


def cliffs(x, y):
    return 2 * mannwhitneyu(x, y, alternative="two-sided").statistic / (len(x) * len(y)) - 1


rows = []
for (m, p), s in df.groupby(["code_module", "code_presentation"]):
    w, f = s[s.final_result == "Withdrawn"], s[s.final_result == "Fail"]
    r = {"course": f"{m}-{p}", "n_withdrawn": len(w), "n_fail": len(f),
         "withdrawn_unreg_on_or_before_day0": int((w.end <= 0).sum())}
    for measure in ["total_clicks", "clicks_per_week"]:
        x, y = w[measure].dropna(), f[measure].dropna()
        r[f"{measure}_median_W"] = round(x.median(), 1) if len(x) else np.nan
        r[f"{measure}_median_F"] = round(y.median(), 1) if len(y) else np.nan
        if len(x) >= 5 and len(y) >= 5:
            r[f"{measure}_p"] = mannwhitneyu(x, y, alternative="two-sided").pvalue
            r[f"{measure}_delta"] = round(cliffs(x, y), 3)
    rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(OUT_FILE, index=False)
pd.set_option("display.width", 220)
print(out[["course", "n_withdrawn", "n_fail", "total_clicks_delta",
           "clicks_per_week_delta"]].to_string(index=False))
print("\nCourses where Withdrawn < Fail (delta < 0):")
print("  total clicks    :", int((out.total_clicks_delta < 0).sum()), "of", len(out))
print("  clicks per week :", int((out.clicks_per_week_delta < 0).sum()), "of", len(out))
print(f"\nSaved: {OUT_FILE}")