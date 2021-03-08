#%%
import numpy as np
from genetic import GeneticAlgorithm
import time
#%%
def fitnessFunction(individual):
    x, y = individual.copy()
    z = np.sin(x)*np.sin(y) + np.sin(y**2)
    return z

#%%
s = time.time()
GA = GeneticAlgorithm(fitnessFunction)
bestCombo = GA.evolution()

print("Execution time: ", time.time() - s)
print(bestCombo)

# %%
