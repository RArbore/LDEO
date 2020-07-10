import matplotlib.pyplot as plt
import math
import sys

comparison_value = int(sys.argv[1])

folders = ["unettrial1", "unettrial2"]

fig = plt.figure()

for i in range(0, len(folders)):
    f = open(folders[i]+"\during_training_performance.txt", "r")
    
    d = []
    #g = []

    line = f.readline()
    while line != "\n" and not "Testing" in line and line != "":
        #d.append(math.log(float(line.split(" ")[1]))-math.log(float(line.split(" ")[2])))
            #g.append(math.log(float(line.split(" ")[2])))
        d.append(float(line.split(" ")[comparison_value]))
        line = f.readline()

    x = range(0, len(d))

    plt.plot(x, d, label=(folders[i]))
    #plt.plot(x, [0]*1000)

    plt.legend()

    #plt.plot(x, d, label=("Discriminator"))

    #plt.legend()

plt.show()
