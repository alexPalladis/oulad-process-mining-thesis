import pickle
import pm4py

# ====== ΑΛΛΑΞΕ ΕΔΩ ΤΙ ΘΕΣ ======
COURSES = [("AAA", "2014J"), ("BBB", "2013J")]
GROUP_PAIRS = [("Pass", "Fail")]   # πρόσθεσε π.χ. ("Distinction", "Fail") αν έχεις τρέξει το discover_all.py με αυτές τις ομάδες
# ================================

MODELS = "../../results/models"

for MODULE, PRES in COURSES:
    with open(f"{MODELS}/models_{MODULE}_{PRES}_noise.pkl", "rb") as f:
        models = pickle.load(f)

    for g1, g2 in GROUP_PAIRS:
        if g1 not in models or g2 not in models:
            print(f"{MODULE}-{PRES}: λείπει μοντέλο για {g1} ή {g2}, παραλείπεται")
            continue

        net1, im1, fm1, log1 = models[g1]["net"], models[g1]["im"], models[g1]["fm"], models[g1]["log"]
        net2, im2, fm2, log2 = models[g2]["net"], models[g2]["im"], models[g2]["fm"], models[g2]["log"]

        print(f"=== {MODULE}-{PRES}: Μοντέλο {g1} πάνω σε log {g2} ===")
        fit = pm4py.fitness_token_based_replay(log2, net1, im1, fm1)
        print(fit)

        print(f"=== {MODULE}-{PRES}: Μοντέλο {g2} πάνω σε log {g1} ===")
        fit2 = pm4py.fitness_token_based_replay(log1, net2, im2, fm2)
        print(fit2)
        print()

print("Ολοκληρώθηκε.")