from numpy import random


class GeneticAlgorithm:
    """ This Class handles all the optimisation for the Genetic Algorithm. The hyperparameters are described below. The default optimisation is for maximisation of the fitnessFunction. For now, the individual is [x, y]
    
        HYPERPARAMETERS: 
        self.populationSize = 1000
        self.nBits = 2
        self.nGenerations = 10
        self.crossoverRate = 0.9
        self.mutatationRate = 0.01 
    """

    def __init__(self, fitnessFunction):

        self.populationSize = 1000
        self.nBits = 2
        self.nGenerations = 10
        self.crossoverRate = 0.9
        self.mutatationRate = 0.01
        self.maxNumber = 2 ** (63) - 1
        self.maxrange = random.default_rng()
        self.fitnessFunction = lambda x: fitnessFunction(x)
        # Population ranges between all possible floats 
        self.population = [
            self.maxrange.uniform(-self.maxNumber, self.maxNumber, size=self.nBits)
            for _ in range(self.populationSize)
        ]
        self.bestChild, self.bestScore = 0, self.fitnessFunction(self.population[0])
        # default is a maximisation problem
        self.maximisation = True

    def evolution(self):

        for generation in range(self.nGenerations):

            print("\n---Generation {} has begun----\n".format(generation + 1))
            scores = [self.fitnessFunction(candidate) for candidate in self.population]

            # The size of the score array and the population should be the same
            assert len(scores) == len(self.population)
            
            # A parent is chosen from 10 random possible parents which has the best 
            # score amongst the random sample
            allParents = [
                self.select_parent(scores) for _ in range(self.populationSize)
            ]
            # Each generation is initialised such that it has no children initiallly
            children = []

            for i in range(self.populationSize):

                if self.maximisation:

                    # Find the best parameters/genes that maximise our fitnessFunction
                    if scores[i] > self.bestScore:
                        self.bestScore, self.bestChild = scores[i], self.population[i]
                        print(
                            "--%g , New best fitness score: %.3f"
                            % (generation + 1, self.bestScore)
                        )
                else:
                    # Find the best parameters/genes that minimise our fitnessFunction
                    if scores[i] < self.bestScore:
                        self.bestScore, self.bestChild = scores[i], self.population[i]
                        print(
                            "--%g , New best fitness score: %.3f"
                            % (generation + 1, self.bestScore)
                        )

            for parentIndex in range(0, self.populationSize - 1, 2):
                
                # Take two adjacent parents and make their children
                parents = allParents[parentIndex : parentIndex + 2]

                for child in self.crossover(parents):

                    # Induce a mutation into the child genes and add it to 
                    # the current generation of children
                    self.mutation(child)
                    children.append(child)

            # Previous generation of population is replaced by the children
            self.population = children

        return self.bestChild

    def mutation(self, individual):

        for gene in range(self.nBits):
            if random.rand() < self.mutatationRate:
                individual[gene] = self.maxrange.uniform(
                    -self.maxNumber, self.maxNumber, size=1
                )[0]

    def crossover(self, parents):

        childOne, childTwo = parents.copy()[0], parents.copy()[1]

        if random.rand() < self.crossoverRate:

            # We take only 1 crossover point for simplicity in our analysis
            # because of which there can be two possible children only
            crossoverPoint = random.randint(0, len(childOne) - 1)

            childOne = [*parents[0][:crossoverPoint], *parents[1][crossoverPoint:]]
            childTwo = [*parents[0][crossoverPoint:], *parents[1][:crossoverPoint]]

        return [childOne, childTwo]

    def select_parent(self, scores):

        # Random sample size to choose a parent form
        access = 10
        selected_index = random.randint(0, self.populationSize)

        for candidate in random.randint(0, self.populationSize, access - 1):

            if self.maximisation:
                if scores[candidate] > scores[selected_index]:
                    selected_index = candidate
            else:
                if scores[candidate] < scores[selected_index]:
                    selected_index = candidate

        # Returns the individual with the best score amongst the chosen sample
        return self.population[selected_index]
