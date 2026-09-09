import pickle
import pm4py

MODULE, PRES = "BBB", "2013J"
with open(f"models_{MODULE}_{PRES}_noise.pkl", "rb") as f:
    models = pickle.load(f)

pass_net, pass_im, pass_fm = models["Pass"]["net"], models["Pass"]["im"], models["Pass"]["fm"]
fail_net, fail_im, fail_fm = models["Fail"]["net"], models["Fail"]["im"], models["Fail"]["fm"]
pass_log = models["Pass"]["log"]
fail_log = models["Fail"]["log"]

print("=== Μοντέλο Pass πάνω σε log Fail (πόσο 'χωράνε' οι Fail στο μοντέλο των Pass) ===")
fit = pm4py.fitness_token_based_replay(fail_log, pass_net, pass_im, pass_fm)
print(fit)

print()
print("=== Μοντέλο Fail πάνω σε log Pass (πόσο 'χωράνε' οι Pass στο μοντέλο των Fail) ===")
fit2 = pm4py.fitness_token_based_replay(pass_log, fail_net, fail_im, fail_fm)
print(fit2)