#%%
import numpy as np
from genetic import GeneticAlgorithm
import time

#%%
def fitnessFunction(individual):
    x, y = individual.copy()
    z = np.sin(x) * np.cos(y) + np.sin(y ** 2) * x
    return z


#%%
startTime = time.time()
GA = GeneticAlgorithm(fitnessFunction)
bestCombo = GA.evolution()

print("Execution time: %.3f" % (time.time() - startTime))
print(bestCombo)

# %%
