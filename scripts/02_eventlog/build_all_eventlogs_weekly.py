import pandas as pd
import os

# --- Ρυθμίσεις διαδρομών (τρέξε το από scripts/02_eventlog/) ---
RAW = "../../data/raw"
OUT = "../../data/processed"
os.makedirs(OUT, exist_ok=True)

svle = pd.read_csv(f"{RAW}/studentVle.csv")
vle = pd.read_csv(f"{RAW}/vle.csv")
info = pd.read_csv(f"{RAW}/studentInfo.csv")

courses = info[["code_module", "code_presentation"]].drop_duplicates().values.tolist()
print(f"Βρέθηκαν {len(courses)} μαθήματα.")

def collapse(g):
    g = g.reset_index(drop=True)
    keep = (g["activity_type"] != g["activity_type"].shift(1))
    return g[keep]

for module, pres in courses:
    vle_c = vle[(vle.code_module == module) & (vle.code_presentation == pres)]
    info_c = info[(info.code_module == module) & (info.code_presentation == pres)]
    svle_c = svle[(svle.code_module == module) & (svle.code_presentation == pres)]

    ev = svle_c.merge(vle_c[["id_site", "activity_type"]], on="id_site", how="left")
    ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")
    # Κρατάμε ΟΛΕΣ τις 4 ομάδες εδώ (Distinction/Pass/Withdrawn/Fail), όχι μόνο Pass/Fail
    ev["week"] = ev["date"] // 7

    weekly = (ev.groupby(["id_student", "final_result", "activity_type", "week"])
              .agg(clicks=("sum_click", "sum")).reset_index())
    weekly = weekly.sort_values(["id_student", "week"])
    weekly = weekly.groupby("id_student", group_keys=False)[weekly.columns].apply(collapse)

    fname = f"{OUT}/event_log_{module}_{pres}_weekly.csv"
    weekly.to_csv(fname, index=False)
    print(f"{module}-{pres}: {weekly.shape[0]} γεγονότα, {weekly.id_student.nunique()} φοιτητές -> {fname}")

print("Ολοκληρώθηκε.")