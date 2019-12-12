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
    fieldnames = ['blurs', 'time', 'std']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(10,510, 10):
        times = []
        for _ in range(50):
            times.append(float(time_prog(i)))
        print("Blurs:", i, statistics.stdev(times), "MEDIAN:", statistics.median(times))
        writer.writerow({'blurs': i, 'time': statistics.median(times), 'std':statistics.stdev(times)})
