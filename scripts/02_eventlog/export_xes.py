import pandas as pd
import pm4py
import os

# ====== ΑΛΛΑΞΕ ΕΔΩ ΤΙ ΘΕΣ ======
MODULE, PRES = "BBB", "2013J"          # ποιο μάθημα
GROUPS = ["Pass", "Fail"]               # ποιες ομάδες (πρόσθεσε "Distinction", "Withdrawn" αν θες)
# ================================

DATA = "../../data/processed"
OUT = "../../data/XES for ProM"
os.makedirs(OUT, exist_ok=True)

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
    sub["time:timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(sub["time:timestamp"] * 7, unit="D")

    log = pm4py.format_dataframe(sub, case_id="case:concept:name",
                                  activity_key="concept:name", timestamp_key="time:timestamp")
    event_log = pm4py.convert_to_event_log(log)

    fname = f"{OUT}/{MODULE}_{PRES}_{outcome}.xes"
    pm4py.write_xes(event_log, fname)
    print(f"{outcome}: {len(event_log)} traces -> {fname}")