import config
import Chromosome
from queue import PriorityQueue

def elitest_survivor(population, population_size=config.POPULATION_SIZE):
    # Min heap to get the N max elements from a list
    minHeap = PriorityQueue()

    for individual in population:
        # Push the individual into the min heap
        minHeap.put(individual)

        # Pop from the min heap if it exceeds the population count
        if minHeap.qsize() > population_size:
            minHeap.get()
    
    # Get the k fittest individuals where k = population_size
    new_population = []
    while not minHeap.empty():
        new_population.append(minHeap.get())

    # If we ever need it, reverse the elites such that they are sorted from best to worst
    new_population.reverse()
    return new_population