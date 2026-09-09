import pandas as pd
import pm4py
import pickle

MODULE, PRES = "BBB", "2013J"
df = pd.read_csv(f"event_log_{MODULE}_{PRES}_weekly.csv")

results = {}
for outcome in ["Pass", "Fail"]:
    sub = df[df.final_result == outcome].copy()
    sub = sub.rename(columns={
        "id_student": "case:concept:name",
        "activity_type": "concept:name",
        "week": "time:timestamp"
    })
    sub["case:concept:name"] = sub["case:concept:name"].astype(str)
    sub["time:timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(sub["time:timestamp"] * 7, unit="D")

    log = pm4py.format_dataframe(sub, case_id="case:concept:name",
                                  activity_key="concept:name",
                                  timestamp_key="time:timestamp")
    event_log = pm4py.convert_to_event_log(log)

    net, im, fm = pm4py.discover_petri_net_inductive(event_log, noise_threshold=0.2)
    print(f"--- {outcome} (noise_threshold=0.2) ---")
    print("Θέσεις:", len(net.places), " Μεταβάσεις:", len(net.transitions))

    pm4py.save_vis_petri_net(net, im, fm, f"petri_{outcome}_{MODULE}_{PRES}_noise.png")

    fitness = pm4py.fitness_token_based_replay(event_log, net, im, fm)
    precision = pm4py.precision_token_based_replay(event_log, net, im, fm)
    print("Fitness:", fitness)
    print("Precision:", precision)
    print()

    results[outcome] = {"net": net, "im": im, "fm": fm, "log": event_log}

with open(f"models_{MODULE}_{PRES}_noise.pkl", "wb") as f:
    pickle.dump(results, f)
print("Έτοιμο.")