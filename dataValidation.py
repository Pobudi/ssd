import json

data = json.load(open("data.json", "r"))

averages = {}

for k, v in data.items() :
    averages[k] = sum(v)/len(v)

json.dump(averages, open("averages.json", "w"))
