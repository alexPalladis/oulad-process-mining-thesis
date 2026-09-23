import pandas as pd
import pm4py
import os

# Ενδεικτική οπτικοποίηση ροής δραστηριοτήτων (directly-follows graph) μέσω PM4Py,
# για το μάθημα AAA-2014J, ξεχωριστά ανά ομάδα έκβασης --- Κεφάλαιο 3.6.
# Σε αντίθεση με το discover_all.py/discover_petri.py (Petri net μέσω Inductive Miner),
# εδώ χρησιμοποιείται ο αλγόριθμος DFG, που δίνει μια απλούστερη, καθαρά περιγραφική
# απεικόνιση της ροής --- χωρίς τυπική εγγύηση soundness.

MODULE, PRES = "AAA", "2014J"
GROUPS = ["Fail", "Pass"]  # σειρά όπως στις Εικόνες 1-2 του Κεφ. 3.6

DATA = "../../data/processed"
FIGS_OUT = "../../results/figures"
os.makedirs(FIGS_OUT, exist_ok=True)

df = pd.read_csv(f"{DATA}/event_log_{MODULE}_{PRES}_weekly.csv")

for outcome in GROUPS:
    sub = df[df.final_result == outcome].copy()
    if len(sub) == 0:
        print(f"{outcome}: καμία εγγραφή, παραλείπεται")
        continue

    sub = sub.rename(columns={
        "id_student": "case:concept:name",
        "activity_type": "concept:name",
        "week": "time:timestamp"
    })
    sub["case:concept:name"] = sub["case:concept:name"].astype(str)
    sub["time:timestamp"] = pd.to_datetime("2014-01-01") + pd.to_timedelta(sub["time:timestamp"] * 7, unit="D")

    log = pm4py.format_dataframe(sub, case_id="case:concept:name",
                                  activity_key="concept:name",
                                  timestamp_key="time:timestamp")
    event_log = pm4py.convert_to_event_log(log)

    dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)

    print(f"--- {outcome} ---")
    print("Ακμές DFG (directly-follows σχέσεις):", len(dfg))
    print("Αρχικές δραστηριότητες:", start_activities)
    print("Τελικές δραστηριότητες:", end_activities)

    img_name = f"{FIGS_OUT}/dfg_{outcome}_{MODULE}_{PRES}.png"
    pm4py.save_vis_dfg(dfg, start_activities, end_activities, img_name)
    print(f"Αποθηκεύτηκε: {img_name}\n")

print("Ολοκληρώθηκε.")