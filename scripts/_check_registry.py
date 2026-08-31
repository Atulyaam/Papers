import json
d = json.load(open("experiments/experiment_registry.json"))
exps = d["experiments"]
for e in exps:
    print(f"{e['id']}: {e['status']}")
print("JSON VALID")
