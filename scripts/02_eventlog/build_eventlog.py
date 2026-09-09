import pandas as pd

MODULE, PRES = "BBB", "2013J"

vle = pd.read_csv("vle.csv")
info = pd.read_csv("studentInfo.csv")
svle = pd.read_csv("studentVle.csv")

vle_c = vle[(vle.code_module == MODULE) & (vle.code_presentation == PRES)]
info_c = info[(info.code_module == MODULE) & (info.code_presentation == PRES)]
svle_c = svle[(svle.code_module == MODULE) & (svle.code_presentation == PRES)]

print("Δραστηριότητες μαθήματος:", vle_c.shape[0])
print("Φοιτητές μαθήματος:", info_c.shape[0])
print(info_c.final_result.value_counts())

ev = svle_c.merge(vle_c[["id_site", "activity_type"]], on="id_site", how="left")
ev = ev.merge(info_c[["id_student", "final_result"]], on="id_student", how="inner")

ev_pf = ev[ev.final_result.isin(["Pass", "Fail"])].copy()
ev_pf["timestamp"] = pd.to_datetime("2014-01-01") + pd.to_timedelta(ev_pf["date"], unit="D")

event_log = (ev_pf.groupby(["id_student", "final_result", "activity_type", "timestamp"])
             .agg(clicks=("sum_click", "sum"))
             .reset_index())

event_log = event_log.sort_values(["id_student", "timestamp"])
event_log.to_csv("event_log_AAA_2014J.csv", index=False)
print("Έτοιμο event log:", event_log.shape)