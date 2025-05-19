import config
from queue import PriorityQueue
import random

def select_elites(population, elite_count=config.ELITE_COUNT):
    """
        This method finds the elites within the population
        Uses min heap to get the N max elements
        
        T.C -> O(n * log k)     n=population size, k = elite_count
        S.C -> O(k)             k = elite_count
    """
    if elite_count <= 0:
        return []
    
    # Min heap to get the N max elements from a list
    minHeap = PriorityQueue()

    for individual in population:
        # Push the individual into the min heap
        minHeap.put(individual)

        # Pop from the min heap if it exceeds the elite count
        if minHeap.qsize() > elite_count:
            minHeap.get()
    
    # Get the k fittest individuals where k = elite_count
    elites = []
    while not minHeap.empty():
        elites.append(minHeap.get())

    # If we ever need it, reverse the elites such that they are sorted from best to worst
    elites.reverse()
    return elites

def spin_the_wheel(population, mating_pool_size=config.MATING_POOL_SIZE):
    """
        T.C. -> O(K * log n)            K = mating_pool_size, n = population size
    """
    # Wheel the roulette K times where K = mating_pool_size (T.C. O(K log n), K=mating_pool_size, n=len(population))
    mating_pool = []
    for _ in range(mating_pool_size):
        # Generate random number
        r = random.random()
        
        # Binary search to find correct individual
        left, right = 0, len(population) - 1
        while left <= right:
            mid = (left + right) // 2
            
            # Find lower bound w_i of current individual
            if mid == 0:
                lower_bound = 0                             # If mid = 0, then we are at the first individual. Its lower bound is of course 0
            else:
                lower_bound = population[mid-1].w_i         # Otherwise start at previous w_i
            
            # If the random number is within the correct range, add the corresponding individual to the pool and break out from the binary search
            # Edge case: if the individual is the first one, no need to control its lower bound since the random number can be 0 even if its very low chance
            if (mid == 0 and r <= population[mid].w_i) or (lower_bound < r <= population[mid].w_i):
                mating_pool.append(population[mid])
                break
            elif r <= lower_bound:      # Search left if the random number is smaller than the lower bound
                right = mid - 1
            else:                       # Search right if the random number is bigger than the upper bound
                left = mid + 1
    return mating_pool

def roulette_wheel_selection(population, mating_pool_size=config.MATING_POOL_SIZE):
    """
        This function implements the roulette wheel selection strategy
        Each individual 'i' gets a protion of the wheel that proportionate to its fitness 'f_i'

            - First, we compute the sum of all the fitnesses
            - Then we calculate p_i (Normalized probability for individual i) for each individual where p_i = f_i / sum_off_fitnesses
            - After that we calculate w_i (cumulative probability) for each individual where w_i = w_i-1 + p_i
            - Finally we wheel the roulette N times where N = mating_pool_size
        
        T.C. -> O(K * log n)            K = mating_pool_size, n = population size
        S.C. -> O(K)                    K = mating_pool_size
    """
    sum_off_fitnesses = 0.0

    # Compute the sum of all the fitnesses (T.C. O(n), n=len(population))
    for individual in population:
        sum_off_fitnesses += individual.fitness

    # Calculate "p_i = f_i / sum_off_fitnesses" for each individual (T.C. O(n), n=len(population))
    for individual in population:
        individual.p_i = individual.fitness / sum_off_fitnesses

    # Calculate w_i for each individual (T.C. O(n), n=len(population))
    population[0].w_i = population[0].p_i

    for i in range(1, len(population)):
        population[i].w_i = population[i-1].w_i + population[i].p_i

    mating_pool = spin_the_wheel(population, mating_pool_size)

    return mating_pool

""" Rank Selection is mostly used when the individuals in the population have very close fitness values  ~ Week 4 lecture notes"""
def rank_selection(population, mating_pool_size=config.MATING_POOL_SIZE):
    """
        T.C. -> O(N * log N)        N=population size
        S.C. -> O(N)                N=population size
    """
    # Sort the population in descending order if it is minimization problem, sort it in ascending order if it is a maximization problem
    if config.IS_MAXIMIZATION:
        sorted_population = sorted(population, key=lambda x: x.fitness)
    else:
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    # Calcualte the cumulative sum of each r_i in the population 
    cumulative_sum_r = len(sorted_population) * (len(sorted_population)+ 1) / 2

    # Assign ranks and cumulative probabilities for each individual
    for i, individual in enumerate(sorted_population):
        individual.r_i = i
        individual.w_i = individual.r_i / cumulative_sum_r

    mating_pool = spin_the_wheel(sorted_population, mating_pool_size)

    return mating_pool

def elitist_selection(population, generation_count, elite_count=config.ELITE_COUNT):
    M_t = []
    
    if generation_count != 1:
        elites = select_elites()
        selections = rank_selection(population, mating_pool_size = config.POPULATION_SIZE-len(elites))
    else:
        elites = []
        selections = rank_selection(population)

    
    
