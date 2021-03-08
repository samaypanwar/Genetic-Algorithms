#%%
import numpy as np
from genetic import GeneticAlgorithm
#%%
def fitnessFunction(individual):
    x, y = individual.copy()
    z = np.sin(x)*np.sin(y) + np.sin(y**2)
    return z

#%%
GA = GeneticAlgorithm(fitnessFunction)
GA.evolution()

