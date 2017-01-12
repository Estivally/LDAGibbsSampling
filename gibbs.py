import random
import math


def gibbs(N=2000, thin=50):
    x = 0
    y = 0
    samples = []
    for i in range(N):
        for j in range(thin):
            x = random.gammavariate(3, 1.0 / (y * y + 4))
            y = random.gauss(1.0 / (x + 1), 1.0 / math.sqrt(x + 1))
        samples.append((x, y))
    return samples

smp = gibbs()
print smp[-500:]
