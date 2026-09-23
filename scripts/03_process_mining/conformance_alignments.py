import pickle
import time
import pandas as pd
import pm4py
from scipy.stats import wilcoxon

# Τρέξε το από τον ίδιο φάκελο με το conformance_all.py (χρησιμοποιεί τα .pkl του discover_all.py)
MODELS = "../../results/models"
OUT = "../../results/statistics"
COURSES = [("AAA", "2014J"), ("BBB", "2013J")]
GROUPS = ["Pass", "Fail"]

rows, tests = [], []
for MODULE, PRES in COURSES:
    with open(f"{MODELS}/models_{MODULE}_{PRES}_noise.pkl", "rb") as f:
        models = pickle.load(f)

    per_trace = {}  # (log_group, model_group) -> λίστα fitness ανά trace
    for log_group in GROUPS:
        for model_group in GROUPS:
            log = models[log_group]["log"]
            net, im, fm = models[model_group]["net"], models[model_group]["im"], models[model_group]["fm"]

            t0 = time.time()
            aligned = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
            fit = [a["fitness"] for a in aligned]
            per_trace[(log_group, model_group)] = fit

            rows.append({
                "course": f"{MODULE}-{PRES}",
                "log": log_group, "model": model_group,
                "type": "own model" if log_group == model_group else "cross",
                "n_traces": len(fit),
                "perc_fit_traces": round(100 * sum(1 for x in fit if x >= 0.999999) / len(fit), 1),
                "avg_trace_fitness": round(sum(fit) / len(fit), 3),
                "seconds": round(time.time() - t0, 1),
            })
            print(rows[-1])

    # Για κάθε log: ίδια traces σε δικό τους μοντέλο έναντι μοντέλου της άλλης ομάδας (paired test)
    for log_group in GROUPS:
        other = [g for g in GROUPS if g != log_group][0]
        own = per_trace[(log_group, log_group)]
        cross = per_trace[(log_group, other)]
        diffs = [c - o for o, c in zip(own, cross)]
        if all(d == 0 for d in diffs):
            p = 1.0
        else:
            p = wilcoxon(own, cross).pvalue
        tests.append({
            "course": f"{MODULE}-{PRES}", "log": log_group,
            "mean_own": round(sum(own) / len(own), 3),
            "mean_cross": round(sum(cross) / len(cross), 3),
            "mean_difference_cross_minus_own": round(sum(diffs) / len(diffs), 3),
            "wilcoxon_p": p,
        })

summary = pd.DataFrame(rows)
paired = pd.DataFrame(tests)
summary.to_csv(f"{OUT}/conformance_alignments.csv", index=False)
paired.to_csv(f"{OUT}/conformance_alignments_paired.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print("\n=== Alignment-based fitness ===")
print(summary.to_string(index=False))
print("\n=== Own vs cross model (same traces, Wilcoxon signed-rank) ===")
print(paired.to_string(index=False))