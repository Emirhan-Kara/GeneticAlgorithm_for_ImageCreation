import Chromosome
import config
import numpy as np
import matplotlib.pyplot as plt
import random
import CNN
from PIL import Image
import evaluation
import selection
import variation
import survivor

# Load the model
config.MODEL = CNN.load_model()

def initialize_population(population_size=config.POPULATION_SIZE):
    """
        This function creates the initial random population of polygon-based chromosomes

        output: population, list containing N chromosomes

        T.C. -> O(n),       n=population_size
        S.C. -> O(n * p),   n=population_size, p=POLYGON_COUNT*POLYGON_PARAMS
    """
    population = []

    # Create random chromosomes (each with random triangles)
    for _ in range(population_size):
        chromosome = Chromosome.Chromosome()  # Will initialize with random gene vector
        population.append(chromosome)
    
    return population

def get_min_and_max_fitness(population):
    """
        This function returns the min and max fitness of the population

        T.C. -> O(n),       n=config.POPULATION_SIZE
        S.C. -> O(1)
    """
    max_fitness = -1
    min_fitness = float('inf')
    for chrom in population:
        if chrom.fitness > max_fitness:
            max_fitness = chrom.fitness
        if chrom.fitness < min_fitness:
            min_fitness = chrom.fitness

    # Return the max and min fitness
    return max_fitness, min_fitness

if __name__ == "__main__":
    population_size = config.POPULATION_SIZE
    elite_count = config.ELITE_COUNT
    mating_pool_size = config.MATING_POOL_SIZE

    # Initialize the population
    population = initialize_population(population_size)

    # Evaluate the population
    elites = []
    evaluation.evaluate_population_with_sharing_function(population)
    
    # Save the best individual's image from the first generation
    best_individual = max(population, key=lambda x: x.fitness)
    best_image = best_individual.reconstruct_image()
    best_image.save(f"best_gen_0.png")
    print(f"Generation 0: Best fitness = {best_individual.fitness}")
    
    for i in range(config.NUM_OF_GENERATIONS):
        best_individual = max(population, key=lambda x: x.fitness)
        print(f"Generation {i+1} best fitness = {best_individual.fitness}")
        if i % config.SAVE_EVERY_N_GENERATIONS == 0:
            best_image = best_individual.reconstruct_image()
            best_image.save(f"best_gen_0.png")

        print("Selection stage...")
        M_t = selection.selection(population, mating_pool_size, elite_count)
        
        print("Variation stage...")
        Q_t = variation.variation(M_t, config.CROSSOVER_RATE, config.MUTATION_RATE)
        R_t = Q_t + elites
        
        print("Evaluation stage...")
        evaluation.evaluate_population_with_sharing_function(R_t)

        print("Survivor stage...")
        population = survivor.elitest_survivor(R_t, population_size)
        elites = selection.select_elites(population, elite_count)
        
        # Save progress every N generations
        if (i+1) % config.SAVE_EVERY_N_GENERATIONS == 0:
            best_individual = max(population, key=lambda x: x.fitness)
            best_image = best_individual.reconstruct_image()
            best_image.save(f"best_gen_{i+1}.png")
            print(f"Generation {i+1}: Best fitness = {best_individual.fitness}")
            
        print("===========================================")

    # Save final best individual
    best_individual = max(population, key=lambda x: x.fitness)
    best_image = best_individual.reconstruct_image()
    best_image.save("final_best.png")
    print(f"Final best fitness: {best_individual.fitness}")
    print(f"Final best objective function: {best_individual.obj_function}")