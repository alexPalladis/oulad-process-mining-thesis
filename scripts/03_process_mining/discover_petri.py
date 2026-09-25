import pandas as pd
import pm4py
import pickle
import os

# Inductive Miner without noise filtering, AAA-2014J (unfiltered models, Table 8).

MODULE, PRES = "AAA", "2014J"

DATA = "../../data/processed"
MODELS_OUT = "../../results/models"
FIGS_OUT = "../../results/figures"
os.makedirs(MODELS_OUT, exist_ok=True)
os.makedirs(FIGS_OUT, exist_ok=True)

df = pd.read_csv(f"{DATA}/event_log_{MODULE}_{PRES}_weekly.csv")

results = {}
for outcome in ["Pass", "Fail"]:
    sub = df[df.final_result == outcome].copy()
    sub = sub.rename(columns={
        "id_student": "case:concept:name",
        "activity_type": "concept:name",
        "week": "time:timestamp"
    })
    sub["case:concept:name"] = sub["case:concept:name"].astype(str)
    # synthetic timestamp from the week number
    sub["time:timestamp"] = pd.to_datetime("2014-01-01") + pd.to_timedelta(sub["time:timestamp"] * 7, unit="D")

    log = pm4py.format_dataframe(sub, case_id="case:concept:name",
                                  activity_key="concept:name",
                                  timestamp_key="time:timestamp")
    event_log = pm4py.convert_to_event_log(log)

    net, im, fm = pm4py.discover_petri_net_inductive(event_log)  # no noise threshold
    print(f"--- {outcome} (no noise filtering) ---")
    print("Θέσεις (places):", len(net.places), " Μεταβάσεις (transitions):", len(net.transitions))

    img_name = f"{FIGS_OUT}/petri_{outcome}_{MODULE}_{PRES}_nonoise.png"
    pm4py.save_vis_petri_net(net, im, fm, img_name)

    fitness = pm4py.fitness_token_based_replay(event_log, net, im, fm)
    precision = pm4py.precision_token_based_replay(event_log, net, im, fm)
    print("Fitness:", fitness)
    print("Precision:", precision)
    print()

    results[outcome] = {"net": net, "im": im, "fm": fm, "log": event_log}

with open(f"{MODELS_OUT}/models_{MODULE}_{PRES}_nonoise.pkl", "wb") as f:
    pickle.dump(results, f)
print("Μοντέλα αποθηκευμένα (χωρίς noise filtering, για σύγκριση με discover_all.py).")