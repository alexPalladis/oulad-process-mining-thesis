import pandas as pd

# --- Ποιο μάθημα να ελέγξουμε (ενδεικτική επικύρωση, όχι πλήρες τρέξιμο) ---
MODULE, PRES = "AAA", "2014J"

DATA = "../../data/processed"
df = pd.read_csv(f"{DATA}/event_log_{MODULE}_{PRES}_daily.csv", parse_dates=["timestamp"])

# Πόσοι φοιτητές, πόσα clicks συνολικά, μέσος όρος ανά φοιτητή
summary = df.groupby("final_result").agg(
    n_students=("id_student", "nunique"),
    total_clicks=("clicks", "sum")
)
summary["avg_clicks_per_student"] = (summary["total_clicks"] / summary["n_students"]).round(1)
print("=== Μέσος όρος clicks ανά φοιτητή ===")
print(summary)

# Μέσος όρος ενεργών ημερών ανά φοιτητή
active_days = df.groupby(["id_student", "final_result"])["timestamp"].nunique().reset_index(name="active_days")
print("\n=== Μέσος όρος ενεργών ημερών ανά φοιτητή ===")
print(active_days.groupby("final_result")["active_days"].mean().round(1))

# Ποσοστό clicks ανά τύπο δραστηριότητας, ανά ομάδα
pct = df.groupby(["final_result", "activity_type"])["clicks"].sum().reset_index()
pct["pct"] = pct.groupby("final_result")["clicks"].transform(lambda x: 100 * x / x.sum())
pivot = pct.pivot(index="activity_type", columns="final_result", values="pct").fillna(0).round(1)
print("\n=== % clicks ανά δραστηριότητα (Pass vs Fail) ===")
print(pivot.sort_values("Pass", ascending=False))