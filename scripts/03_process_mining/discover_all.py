import pandas as pd
import pm4py
import pickle
import os

COURSES = [("AAA", "2014J"), ("BBB", "2013J")]   
GROUPS = ["Pass", "Fail"]                          
NOISE_THRESHOLD = 0.2

DATA = "../../data/processed"
MODELS_OUT = "../../results/models"
FIGS_OUT = "../../results/figures"
os.makedirs(MODELS_OUT, exist_ok=True)
os.makedirs(FIGS_OUT, exist_ok=True)

for MODULE, PRES in COURSES:
    df = pd.read_csv(f"{DATA}/event_log_{MODULE}_{PRES}_weekly.csv")
    results = {}

    for outcome in GROUPS:
        sub = df[df.final_result == outcome].copy()
        if len(sub) < 10:
            print(f"{MODULE}-{PRES} {outcome}: πολύ λίγα δεδομένα, παραλείπεται")
            continue

        sub = sub.rename(columns={
            "id_student": "case:concept:name",
            "activity_type": "concept:name",
            "week": "time:timestamp"
        })
        sub["case:concept:name"] = sub["case:concept:name"].astype(str)
        sub["time:timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(sub["time:timestamp"] * 7, unit="D")

        log = pm4py.format_dataframe(sub, case_id="case:concept:name",
                                      activity_key="concept:name", timestamp_key="time:timestamp")
        event_log = pm4py.convert_to_event_log(log)

        print(f"--- {MODULE}-{PRES} / {outcome} ({len(event_log)} traces) ---")
        net, im, fm = pm4py.discover_petri_net_inductive(event_log, noise_threshold=NOISE_THRESHOLD)
        print("Θέσεις:", len(net.places), " Μεταβάσεις:", len(net.transitions))

        img_name = f"{FIGS_OUT}/petri_{outcome}_{MODULE}_{PRES}_noise.png"
        pm4py.save_vis_petri_net(net, im, fm, img_name)

        fitness = pm4py.fitness_token_based_replay(event_log, net, im, fm)
        precision = pm4py.precision_token_based_replay(event_log, net, im, fm)
        print("Fitness:", fitness)
        print("Precision:", precision)
        print()

        results[outcome] = {"net": net, "im": im, "fm": fm, "log": event_log}

    with open(f"{MODELS_OUT}/models_{MODULE}_{PRES}_noise.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Αποθηκεύτηκαν τα μοντέλα για {MODULE}-{PRES}\n")

print("Ολοκληρώθηκε.")