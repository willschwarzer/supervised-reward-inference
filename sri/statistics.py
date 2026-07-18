import numpy as np
import random
from tqdm import tqdm

def m(z, u):
    n = len(z)
    result = 1.0
    for i in range(n - 1):
        result -= (z[i + 1] - z[i]) * u[i]
    result -= (1.0 - z[n - 1]) * u[n - 1]
    return result

def Gaffke(x, delta, UPrimes):
    x = sorted(x)
    ms = [m(x, u) for u in UPrimes]
    ms.sort(reverse=True)
    return ms[int(delta * len(ms))]

def loadUPrimes(nUPrimes, n):
    UPrimes = []
    for _ in range(nUPrimes):
        u = sorted([random.uniform(0, 1) for _ in range(n)])
        UPrimes.append(u)
    return UPrimes

def compute_gaffke_ci(sample, delta, nUPrimes):
    n = len(sample)
    UPrimes = loadUPrimes(nUPrimes, n)
    # Upper bound calculation
    upperBound = Gaffke(sample, delta, UPrimes)
    
    # Lower bound calculation (transform, compute, undo transformation)
    transformedSample = [-x + 1 for x in sample]
    lowerBound = 1 - Gaffke(transformedSample, delta, UPrimes)
    
    return lowerBound, upperBound

def main():
    nUPrimes = 10000
    n = 20
    numTrials = 1000
    alpha, beta = 5.0, 1.0
    delta = 0.05
    mean = alpha / (alpha + beta)
    
    numCorrect = 0  # Tracks the number of trials where the mean is within the computed bounds

    for _ in tqdm(range(numTrials)):
        sample = np.random.beta(alpha, beta, n)
        lowerBound, upperBound = compute_gaffke_ci(sample, delta, nUPrimes)
        
        # Check if the true mean is within the computed bounds
        if lowerBound <= mean <= upperBound:
            numCorrect += 1
    
    correctnessProbability = numCorrect / numTrials

    print(f"We ran {numTrials} trials with n = {n} samples from Beta({alpha}, {beta}) distribution and delta = {delta}.")
    print(f"\tActual mean: {mean}")
    print(f"\tComputed bounds: [{lowerBound:.2f}, {upperBound:.2f}]")
    print(f"\tCorrectness of the computed bounds across all trials: {correctnessProbability:.2f}")

if __name__ == "__main__":
    main()
