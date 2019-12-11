import os
import re
import statistics
import csv



def time_prog(blurs):
    o = os.popen("./blur " + str(blurs))
    s = o.read()
    matches = re.findall('\d*\.\d*', s)
    return matches[0]

os.system("nvcc -o blur blur.cu")
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['blurs', 'time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1,100):
        times = []
        for _ in range(10):
            times.append(float(time_prog(i)))
        print("Blurs:", i, statistics.stdev(times), "MEAN:", statistics.mean(times))
        writer.writerow({'blurs': i, 'time': statistics.mean(times)})
