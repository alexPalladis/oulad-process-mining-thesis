import pandas as pd
from scipy import stats
from scipy.stats import rankdata

df = pd.read_csv("event_log_AAA_2014J.csv", parse_dates=["timestamp"])

per_student = df.groupby(["id_student", "final_result"])["clicks"].sum().reset_index()

fail_clicks = per_student[per_student.final_result == "Fail"]["clicks"]
pass_clicks = per_student[per_student.final_result == "Pass"]["clicks"]

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    ranks = rankdata(list(x) + list(y))
    rx = sum(ranks[:nx])
    auc = (rx - nx*(nx+1)/2) / (nx*ny)
    return 2*auc - 1

t, p = stats.mannwhitneyu(fail_clicks, pass_clicks)
print("Mann-Whitney U test (clicks ανά φοιτητή, Pass vs Fail)")
print("p-value:", p)
print("Cliff's delta:", cliffs_delta(pass_clicks, fail_clicks))