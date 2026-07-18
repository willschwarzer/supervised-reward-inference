import os

os.mkdir('example_size_run')

for i in [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
    os.system("python3.6 predict_transformer.py " + str(i) + " > example_size_run/" + str(i) + ".txt")
