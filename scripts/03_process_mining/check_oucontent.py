import pickle
import numpy as np
from scipy.stats import chi2_contingency

m = pickle.load(open("../../results/models/models_BBB_2013J_noise.pkl", "rb"))

counts = {}
for g in ["Pass", "Fail"]:
    log = m[g]["log"]
    without = sum(1 for t in log if not any(e["concept:name"] == "oucontent" for e in t))
    counts[g] = (without, len(log) - without)
    print(g, without, "/", len(log), "traces χωρίς oucontent", f"({100 * without / len(log):.1f}%)")

table = np.array([counts["Pass"], counts["Fail"]])
chi2, p, dof, _ = chi2_contingency(table)
phi = np.sqrt(chi2 / table.sum())
print(f"chi2 = {chi2:.1f}, dof = {dof}, p = {p:.2e}, phi = {phi:.3f}")