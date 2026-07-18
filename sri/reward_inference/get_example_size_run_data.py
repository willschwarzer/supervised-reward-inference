import os

data = os.listdir("example_size_run")

for run in data:
    with open("example_size_run/" + run, "r") as f:
        pn = False
        val = float("-inf")
        for line in f:
            if pn:
                val = max(val, float(line.strip().split("val acc")[1].strip()))
                pn = False
            if ord(line[0]) == 10:
                pn = True
    print(run[:-4] + ",", val)
