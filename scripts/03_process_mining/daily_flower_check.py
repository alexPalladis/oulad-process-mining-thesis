"""Robustness check for Section 4.2: rediscovers the Pass and Fail models from a
daily event log (same-day order randomized) and computes the heuristic
dependency measure for transitions between different days.
Output: results/process_mining/daily_check/"""
import os
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.process_tree.obj import Operator

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = None    
OUT_DIR = REPO_ROOT / "results" / "process_mining" / "daily_check"

COURSES = [("AAA", "2014J"), ("BBB", "2013J")]
GROUPS = ["Pass", "Fail"]
NOISE = [0.0, 0.1, 0.2, 0.3]
FIG_NOISE = 0.2
SEED = 42
MIN_PAIR_SUPPORT = 30     
STRONG_DEP = 0.5          
CHUNK_SIZE = 2_000_000
BASE_DATE = pd.Timestamp("2014-01-01")

if RAW_DIR is None:
    hits = [p.parent for p in REPO_ROOT.rglob("studentInfo.csv") if "results" not in p.parts]
    if not hits:
        sys.exit(f"studentInfo.csv not found under {REPO_ROOT}. Set RAW_DIR.")
    RAW_DIR = hits[0]
RAW_DIR = Path(RAW_DIR)
print(f"Raw data folder: {RAW_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)

INFO = pd.read_csv(RAW_DIR / "studentInfo.csv")
VLE = pd.read_csv(RAW_DIR / "vle.csv")
wanted = {f"{m}|{p}" for m, p in COURSES}
parts = []
for ch in pd.read_csv(RAW_DIR / "studentVle.csv", chunksize=CHUNK_SIZE):
    ch = ch[(ch.code_module + "|" + ch.code_presentation).isin(wanted)]
    if len(ch):
        parts.append(ch)
SVLE = pd.concat(parts, ignore_index=True)


def in_course(df, m, p):
    return df[(df.code_module == m) & (df.code_presentation == p)]


def build_daily_log(m, p):
    info = in_course(INFO, m, p)
    info = info[info.final_result.isin(GROUPS)][["id_student", "final_result"]]
    vle = in_course(VLE, m, p)[["id_site", "activity_type"]]
    s = in_course(SVLE, m, p).merge(vle, on="id_site", how="left")
    s = s.merge(info, on="id_student", how="inner")
    ev = (s.groupby(["id_student", "final_result", "activity_type", "date"], as_index=False)
            ["sum_click"].sum())
    rng = np.random.default_rng(SEED)
    ev["rnd"] = rng.random(len(ev))                     # τυχαία σειρά μέσα στη μέρα
    ev = ev.sort_values(["id_student", "date", "rnd"]).reset_index(drop=True)
    ev["rank"] = ev.groupby(["id_student", "date"]).cumcount()
    ev["timestamp"] = (BASE_DATE + pd.to_timedelta(ev.date, unit="D")
                       + pd.to_timedelta(ev["rank"], unit="s"))
    ev["case_id"] = ev.id_student.astype(str)
    return ev.drop(columns=["rnd", "rank"])


def tree_stats(tree):
    """Μετράει τελεστές και πόσες δραστηριότητες είναι 'βρόχος μέσα σε parallel'."""
    ops = {"seq": 0, "par": 0, "loop": 0, "xor": 0}
    labels, flower = set(), set()

    def visit(node, parent=None):
        if node.operator is None:
            if node.label:
                labels.add(node.label)
                if parent is not None and parent.operator == Operator.PARALLEL:
                    flower.add(node.label)
            return
        op = node.operator
        if op == Operator.SEQUENCE: ops["seq"] += 1
        elif op == Operator.PARALLEL: ops["par"] += 1
        elif op == Operator.LOOP: ops["loop"] += 1
        elif op == Operator.XOR: ops["xor"] += 1
        if op == Operator.LOOP and parent is not None and parent.operator == Operator.PARALLEL:
            body = node.children[0] if node.children else None
            if body is not None and body.operator is None and body.label:
                flower.add(body.label)
        for c in node.children:
            visit(c, node)

    visit(tree)
    share = round(len(flower) / len(labels), 3) if labels else None
    return ops, len(labels), share


def order_consistency(ev):
    """Dependency measure μόνο για διαδοχές ανάμεσα σε διαφορετικές ημέρες."""
    counts = {}
    for _, g in ev.groupby("id_student"):
        days = g.groupby("date")["activity_type"].apply(set).sort_index()
        prev = None
        for acts in days.values:
            if prev is not None:
                for a, b in product(prev, acts):
                    if a != b:
                        counts[(a, b)] = counts.get((a, b), 0) + 1
            prev = acts
    rows, seen = [], set()
    for (a, b), n_ab in counts.items():
        if (b, a) in seen:
            continue
        seen.add((a, b))
        n_ba = counts.get((b, a), 0)
        if n_ab + n_ba < MIN_PAIR_SUPPORT:
            continue
        dep = (n_ab - n_ba) / (n_ab + n_ba + 1)
        rows.append({"a": a, "b": b, "a_then_b": n_ab, "b_then_a": n_ba, "dep": round(dep, 3)})
    return pd.DataFrame(rows)


model_rows, order_rows, pair_frames, trees = [], [], [], []
for m, p in COURSES:
    course = f"{m}-{p}"
    print(f"\n######## {course} ########")
    log = build_daily_log(m, p)
    for g in GROUPS:
        ev = log[log.final_result == g]
        df = pm4py.format_dataframe(ev, case_id="case_id", activity_key="activity_type",
                                    timestamp_key="timestamp")
        n_tr = df["case:concept:name"].nunique()
        avg_len = round(len(df) / n_tr, 1)
        print(f"{g}: {n_tr} traces, {len(df)} events, {avg_len} events/trace")
        for t in NOISE:
            tree = pm4py.discover_process_tree_inductive(df, noise_threshold=t)
            net, im, fm = pm4py.discover_petri_net_inductive(df, noise_threshold=t)
            ops, n_act, share = tree_stats(tree)
            model_rows.append({"course": course, "group": g, "noise": t, "traces": n_tr,
                               "events_per_trace": avg_len, "activities": n_act,
                               "places": len(net.places), "transitions": len(net.transitions),
                               "n_seq": ops["seq"], "n_par": ops["par"],
                               "n_loop": ops["loop"], "n_xor": ops["xor"],
                               "flower_share": share})
            trees.append(f"=== {course} / {g} / noise {t} ===\n{tree}\n")
            print(f"  noise={t}: flower_share={share}, seq={ops['seq']}, par={ops['par']}, "
                  f"loop={ops['loop']}, xor={ops['xor']}")
            if t == FIG_NOISE:
                try:
                    pm4py.save_vis_petri_net(net, im, fm,
                                             str(OUT_DIR / f"daily_model_{course}_{g}.png"))
                except Exception as e:
                    print(f"  [warn] figure not saved: {e}")

        pairs = order_consistency(ev)
        pairs.insert(0, "group", g); pairs.insert(0, "course", course)
        pair_frames.append(pairs)
        a = pairs.dep.abs()
        order_rows.append({"course": course, "group": g, "pairs_tested": len(pairs),
                           "mean_abs_dep": round(a.mean(), 3) if len(a) else None,
                           "max_abs_dep": round(a.max(), 3) if len(a) else None,
                           "pairs_strong_order": int((a >= STRONG_DEP).sum()),
                           "share_strong_order": round((a >= STRONG_DEP).mean(), 3) if len(a) else None})

ms = pd.DataFrame(model_rows); ms.to_csv(OUT_DIR / "daily_models_summary.csv", index=False)
os_ = pd.DataFrame(order_rows); os_.to_csv(OUT_DIR / "daily_order_summary.csv", index=False)
pd.concat(pair_frames, ignore_index=True).to_csv(OUT_DIR / "daily_order_pairs.csv", index=False)
with open(OUT_DIR / "daily_trees.txt", "w", encoding="utf-8") as f:
    f.write("Operators: -> sequence, X choice, + parallel, * loop, tau silent\n\n")
    f.write("\n".join(trees))

pd.set_option("display.width", 200)
print("\n=== ΔΟΜΗ ΜΟΝΤΕΛΩΝ (ημερήσιο log) ===")
print(ms.to_string(index=False))
print("\n=== ΣΤΑΘΕΡΟΤΗΤΑ ΣΕΙΡΑΣ (μόνο διαδοχές σε διαφορετικές ημέρες) ===")
print(os_.to_string(index=False))
print(f"\n|dep| >= {STRONG_DEP}: σταθερή σειρά. Κοντά στο 0: καμία σταθερή σειρά.")
print(f"Done. Outputs in {OUT_DIR}")