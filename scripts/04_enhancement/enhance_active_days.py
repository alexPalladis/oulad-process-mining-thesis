"""Chapter 6, step 2: active days per student and phase (phases delimited by TMA
deadlines; students included only if registered for the whole phase).
Pass vs Fail and Withdrawn vs Fail: Mann-Whitney U, Cliff's delta, Holm per course.
Output: results/enhancement/"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = None      
OUT_DIR = REPO_ROOT / "results" / "enhancement"

COURSES = [("AAA", "2014J"), ("BBB", "2013J")]
GROUPS = ["Pass", "Fail", "Withdrawn"]
PAIRS = [("Pass", "Fail"), ("Withdrawn", "Fail")]
LAST_DAYS = 7
ALPHA = 0.05
MIN_N = 5                               
CHUNK_SIZE = 2_000_000
COLORS = {"Pass": "#2b6cb0", "Fail": "#c53030", "Withdrawn": "#718096"}
MARKERS = {"Pass": "o", "Fail": "s", "Withdrawn": "^"}


if RAW_DIR is None:
    hits = [p.parent for p in REPO_ROOT.rglob("studentInfo.csv") if "results" not in p.parts]
    if not hits:
        sys.exit(f"studentInfo.csv not found under {REPO_ROOT}. Set RAW_DIR.")
    RAW_DIR = hits[0]
RAW_DIR = Path(RAW_DIR)
print(f"Raw data folder: {RAW_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)

INFO = pd.read_csv(RAW_DIR / "studentInfo.csv")
REG = pd.read_csv(RAW_DIR / "studentRegistration.csv", na_values=["?", ""])
ASSESS = pd.read_csv(RAW_DIR / "assessments.csv", na_values=["?", ""])
SA = pd.read_csv(RAW_DIR / "studentAssessment.csv", na_values=["?", ""])
wanted = {f"{m}|{p}" for m, p in COURSES}
parts = []
for ch in pd.read_csv(RAW_DIR / "studentVle.csv", chunksize=CHUNK_SIZE):
    ch = ch[(ch.code_module + "|" + ch.code_presentation).isin(wanted)]
    if len(ch):
        parts.append(ch[["code_module", "code_presentation", "id_student", "date", "sum_click"]])
SVLE = pd.concat(parts, ignore_index=True)


def in_course(df, m, p):
    return df[(df.code_module == m) & (df.code_presentation == p)]


def compare(x, y):
    """Mann-Whitney U (two-sided) and Cliff's delta = 2U/(n1*n2) - 1."""
    if len(x) < MIN_N or len(y) < MIN_N:
        return np.nan, np.nan, np.nan
    res = mannwhitneyu(x, y, alternative="two-sided")
    return res.statistic, res.pvalue, 2 * res.statistic / (len(x) * len(y)) - 1


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def run_tests(phases, measures, course):
    rows = []
    for ph in phases.phase.unique():
        s = phases[phases.phase == ph]
        for meas in measures:
            for a, b in PAIRS:
                x = s[s.final_result == a][meas].dropna()
                y = s[s.final_result == b][meas].dropna()
                u, pv, d = compare(x, y)
                rows.append({"course": course, "phase": ph, "measure": meas,
                             "comparison": f"{a} vs {b}", "n1": len(x), "n2": len(y),
                             "median1": x.median(), "median2": y.median(),
                             "U": u, "p": pv, "cliffs_delta": round(d, 3)})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t["significant_holm"] = t.p_holm < ALPHA
    return t


desc, main_tests, supp_tests, profiles = [], [], [], {}

for m, p in COURSES:
    course = f"{m}-{p}"
    print(f"\n######## {course} ########")
    info = in_course(INFO, m, p)
    info = info[info.final_result.isin(GROUPS)][["id_student", "final_result"]]
    reg = in_course(REG, m, p)[["id_student", "date_registration", "date_unregistration"]]
    vle = in_course(SVLE, m, p)
    vle = vle[vle.id_student.isin(info.id_student) & (vle.sum_click > 0)]
    daily = vle.groupby(["id_student", "date"], as_index=False).sum_click.sum()

    a = in_course(ASSESS, m, p)
    a = a[(a.assessment_type == "TMA") & a.date.notna()].copy()
    a["date"] = a.date.astype(int)
    tma = a.sort_values("date").reset_index(drop=True)
    sa = SA[(SA.is_banked == 0) & SA.id_student.isin(info.id_student)
            & SA.id_assessment.isin(in_course(ASSESS, m, p).id_assessment)]

    ids = set(daily.id_student) | set(sa.id_student)
    stud = info[info.id_student.isin(ids)].merge(reg, on="id_student", how="left")
    print("students per group:", stud.final_result.value_counts().to_dict())

    rows, start = [], 0
    for j, end in enumerate(tma.date, start=1):
        ok = ((stud.date_registration.isna() | (stud.date_registration <= end)) &
              (stud.date_unregistration.isna() | (stud.date_unregistration >= end)))
        elig = stud[ok]
        d = daily[(daily.date >= start) & (daily.date <= end) & daily.id_student.isin(elig.id_student)]
        g = d.groupby("id_student")
        agg = pd.DataFrame({"active_days": g.date.nunique(),
                            "first_day": g.date.min(),
                            "clicks": g.sum_click.sum(),
                            "clicks_last": d[d.date >= end - LAST_DAYS + 1]
                                           .groupby("id_student").sum_click.sum()})
        ph = elig[["id_student", "final_result"]].set_index("id_student").join(agg)
        ph["active_days"] = ph.active_days.fillna(0).astype(int)
        ph["days_to_first_activity"] = ph.first_day - start
        ph["last7_share"] = ph.clicks_last.fillna(0) / ph.clicks
        ph["phase"], ph["phase_start"], ph["phase_end"] = f"P{j}", start, end
        ph["phase_length"] = end - start + 1
        rows.append(ph.reset_index())
        start = end + 1
    phases = pd.concat(rows, ignore_index=True)
    phases.drop(columns=["first_day", "clicks", "clicks_last"]).to_csv(
        OUT_DIR / f"active_days_per_student_{course}.csv", index=False)

    for (ph, grp), s in phases.groupby(["phase", "final_result"]):
        desc.append({"course": course, "phase": ph, "phase_length": s.phase_length.iloc[0],
                     "group": grp, "n_registered": len(s),
                     "pct_inactive": round(100 * (s.active_days == 0).mean(), 1),
                     "median_active_days": s.active_days.median(),
                     "iqr_active_days": f"{s.active_days.quantile(.25):g}-{s.active_days.quantile(.75):g}"})
    profiles[course] = phases.groupby(["phase", "final_result"]).active_days.median().unstack()

    main_tests.append(run_tests(phases, ["active_days"], course))
    supp_tests.append(run_tests(phases, ["days_to_first_activity", "last7_share"], course))


pd.DataFrame(desc).to_csv(OUT_DIR / "active_days_descriptives.csv", index=False)
mt = pd.concat(main_tests, ignore_index=True).drop(columns="measure")
mt.to_csv(OUT_DIR / "active_days_tests.csv", index=False)
st = pd.concat(supp_tests, ignore_index=True)
st.to_csv(OUT_DIR / "supplementary_timing_tests.csv", index=False)

fig, axes = plt.subplots(1, len(COURSES), figsize=(6 * len(COURSES), 4.2), sharey=False)
axes = np.atleast_1d(axes)
for ax, (course, prof) in zip(axes, profiles.items()):
    x = range(1, len(prof) + 1)
    for grp in GROUPS:
        if grp in prof:
            ax.plot(x, prof[grp].values, marker=MARKERS[grp], color=COLORS[grp], label=grp)
    ax.set_xticks(list(x))
    ax.set_xticklabels(prof.index)
    ax.set_title(course)
    ax.set_xlabel("Phase (ends at the deadline of TMA j)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Median active days per student")
axes[0].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "active_days_per_phase.png", dpi=300)
plt.close(fig)

pd.set_option("display.width", 200)
print("\nMain table (active days):")
print(mt[["course", "phase", "comparison", "n1", "n2", "median1", "median2",
          "cliffs_delta", "p_holm"]].to_string(index=False))
print("\nSupplementary timing measures -- Holm-significant results:")
sig = st[st.significant_holm]
print(sig[["course", "phase", "measure", "comparison", "cliffs_delta", "p_holm"]]
      .to_string(index=False) if len(sig) else "  none")
print(f"\nDone. Outputs in {OUT_DIR}")