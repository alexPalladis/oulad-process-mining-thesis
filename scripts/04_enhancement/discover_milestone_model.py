"""
discover_milestone_model.py  --  Chapter 6, step 1 of 2

Builds the enriched event log of each course and discovers the process model
of Pass students that is enhanced in Chapter 6.

Enriched event log (differences from the Chapter 4-5 log)
---------------------------------------------------------
  - VLE events per (student, activity_type, week), as in Chapter 4, but
    consecutive repetitions are NOT collapsed.
  - Event timestamp = first day on which that activity type was accessed in
    that week (so the order within a week is no longer arbitrary).
  - Assessment submissions are added as milestone events "submit_<TYPE><n>"
    (n = order by deadline within the type). Banked submissions and exams
    (no deadline date in assessments.csv) are excluded.
  - Same-day ties: VLE events before submissions, then alphabetical.

Model
-----
Inductive Miner (PM4Py) on the Pass log, noise threshold 0.2 as in Chapter 4.
The TMA milestones of this model delimit the phases used in step 2
(enhance_active_days.py). As a robustness check, the script verifies that all
TMA milestones remain in the Pass model at noise thresholds 0.0, 0.1 and 0.2.

Outputs (results/enhancement/)
------------------------------
  enriched_log_<course>.csv
  pass_model_<course>.png            Petri net of the Pass model (noise 0.2)
  pass_model_trees.txt               process trees, all thresholds
  milestone_check.csv                TMA milestones present per threshold
"""

import os
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
import pm4py

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = None      # e.g. Path(r"C:\Users\Alex\Desktop\thesis_oulad\data\raw")
OUT_DIR = REPO_ROOT / "results" / "enhancement"

COURSES = [("AAA", "2014J"), ("BBB", "2013J")]
GROUPS = ["Pass", "Fail", "Withdrawn"]    # all groups are kept in the log
MODEL_NOISE = 0.2                         # threshold of the reported model
CHECK_NOISE = [0.0, 0.1, 0.2]             # robustness check of the milestones
BASE_DATE = dt.datetime(2014, 1, 1)       # synthetic day 0 (pm4py needs dates)
CHUNK_SIZE = 2_000_000

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
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
VLE = pd.read_csv(RAW_DIR / "vle.csv")
ASSESS = pd.read_csv(RAW_DIR / "assessments.csv", na_values=["?", ""])
SA = pd.read_csv(RAW_DIR / "studentAssessment.csv", na_values=["?", ""])
wanted = {f"{m}|{p}" for m, p in COURSES}
parts = []
for ch in pd.read_csv(RAW_DIR / "studentVle.csv", chunksize=CHUNK_SIZE):
    ch = ch[(ch.code_module + "|" + ch.code_presentation).isin(wanted)]
    if len(ch):
        parts.append(ch)
SVLE = pd.concat(parts, ignore_index=True)


def in_course(df, m, p):
    return df[(df.code_module == m) & (df.code_presentation == p)]


def milestones_of(m, p):
    a = in_course(ASSESS, m, p)
    a = a[(a.assessment_type != "Exam") & a.date.notna()].copy()
    a["date"] = a.date.astype(int)
    a = a.sort_values(["date", "assessment_type", "id_assessment"])
    a["seq"] = a.groupby("assessment_type").cumcount() + 1
    a["activity"] = "submit_" + a.assessment_type + a.seq.astype(str)
    return a


def build_log(m, p):
    info = in_course(INFO, m, p)
    info = info[info.final_result.isin(GROUPS)][["id_student", "final_result"]]
    reg = in_course(REG, m, p)[["id_student", "date_unregistration"]]
    vle = in_course(VLE, m, p)[["id_site", "activity_type"]]
    assess = milestones_of(m, p)

    svle = in_course(SVLE, m, p).merge(vle, on="id_site", how="left")
    svle = svle[svle.id_student.isin(info.id_student)].copy()
    svle["week"] = svle.date // 7
    ve = (svle.groupby(["id_student", "activity_type", "week"], as_index=False)
              .agg(day=("date", "min"), clicks=("sum_click", "sum"))
              .rename(columns={"activity_type": "activity"}))
    ve["event_kind"], ve["sort_priority"] = "vle", 0

    sa = SA[(SA.is_banked == 0) & SA.id_assessment.isin(assess.id_assessment)
            & SA.id_student.isin(info.id_student)]
    sa = sa.merge(assess[["id_assessment", "activity", "date"]], on="id_assessment")
    sa = sa.rename(columns={"date": "deadline", "date_submitted": "day"})
    sa["week"] = sa.day // 7
    sa["event_kind"], sa["sort_priority"] = "submission", 1
    se = sa[["id_student", "activity", "week", "day", "deadline", "event_kind", "sort_priority"]]

    log = pd.concat([ve, se], ignore_index=True)
    log = log.merge(info, on="id_student").merge(reg, on="id_student", how="left")
    log = log.sort_values(["id_student", "day", "sort_priority", "activity"]).reset_index(drop=True)
    log["rank_in_day"] = log.groupby(["id_student", "day"]).cumcount()
    log["timestamp"] = (BASE_DATE + pd.to_timedelta(log.day, unit="D")
                        + pd.to_timedelta(log.rank_in_day, unit="s"))
    log["case_id"] = log.id_student.astype(str)
    return log.drop(columns=["sort_priority", "rank_in_day"]), assess


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
check_rows, trees = [], []
for m, p in COURSES:
    course = f"{m}-{p}"
    print(f"\n######## {course} ########")
    log, assess = build_log(m, p)
    log.to_csv(OUT_DIR / f"enriched_log_{course}.csv", index=False)
    tmas = list(assess[assess.assessment_type == "TMA"].activity)

    df = pm4py.format_dataframe(log[log.final_result == "Pass"], case_id="case_id",
                                activity_key="activity", timestamp_key="timestamp")
    el = pm4py.convert_to_event_log(df)

    for t in CHECK_NOISE:
        tree = pm4py.discover_process_tree_inductive(el, noise_threshold=t)
        net, im, fm = pm4py.discover_petri_net_inductive(el, noise_threshold=t)
        labels = {tr.label for tr in net.transitions if tr.label}
        present = [a for a in tmas if a in labels]
        row = {"course": course, "noise": t, "traces": len(el),
               "tma_milestones_total": len(tmas),
               "tma_milestones_in_model": len(present),
               "all_tma_present": len(present) == len(tmas),
               "places": len(net.places), "transitions": len(net.transitions)}
        if t == MODEL_NOISE:
            fit = pm4py.fitness_token_based_replay(el, net, im, fm)
            row["fitness_tbr_avg"] = round(fit["average_trace_fitness"], 3)
            try:
                pm4py.save_vis_petri_net(net, im, fm, str(OUT_DIR / f"pass_model_{course}.png"))
            except Exception as e:
                print(f"[warn] figure not saved: {e}")
        check_rows.append(row)
        trees.append(f"=== {course} / Pass / noise {t} ===\n{tree}\n")
        print(f"noise={t}: {len(present)}/{len(tmas)} TMA milestones in the Pass model")

chk = pd.DataFrame(check_rows)
chk.to_csv(OUT_DIR / "milestone_check.csv", index=False)
with open(OUT_DIR / "pass_model_trees.txt", "w", encoding="utf-8") as f:
    f.write("Operators: -> sequence, X choice, + parallel, * loop, tau silent\n\n")
    f.write("\n".join(trees))
if not chk.all_tma_present.all():
    print("\n[!] Not all TMA milestones are present in every Pass model -- see milestone_check.csv")
print(f"\nDone. Outputs in {OUT_DIR}")