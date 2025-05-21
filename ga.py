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

def initialize_population(population_size=config.POPULATION_SIZE, image_size=config.IMAGE_SIZE, channel_num=config.CHANNEL_NUM):
    """
        This function creates the initial random population

        output: populatiion, list containing N chromosomes

        T.C. -> O(n * m),       n=population_size, m=RANDOM_IMAGE_CREATION_ITERATIONS
        S.C. -> O(n * m),       n=population_size, m=image_size ^2 *  channel_num
    """
    population = []

    # This loop creates the initial random population
    # Each chromosome is derived by first creating a random image.
    # Then a rectangle area within the image is randomly selected at each iteration and a random color is added to the area
    for _ in range(population_size):
        # Create a random RGB color for the background
        random_color = np.random.rand(3)

        # Generate a random vector with the desired size to represent the random image 
        random_image_vector = np.full((image_size, image_size, channel_num), random_color)

        # This loop iterates over the same random image and at each iteration it adds  a random color to the area
        # The rectangle is defined by the two random points
        for _ in range(config.RANDOM_IMAGE_CREATION_ITERATIONS):
            # Create 2 random points in the image
            # These points will be used to create a rectangle
            random_x1 = random.randint(0, image_size - 1)
            random_x2 = random.randint(0, image_size - 1)
            random_y1 = random.randint(0, image_size - 1)    
            random_y2 = random.randint(0, image_size - 1)

            # Calculate the north-west corner of the rectangle
            x1 = min(random_x1, random_x2)
            y1 = min(random_y1, random_y2)

            # Calculate the south-east corner of the rectangle
            x2 = max(random_x1, random_x2)
            y2 = max(random_y1, random_y2)

            # Create a random RGB color
            random_color = np.random.rand(3)

            # Fill the rectangle by summing the random color with the existing pixel values
            random_image_vector[x1:x2, y1:y2, :] = (random_image_vector[x1:x2, y1:y2, :] + random_color) % 1.0

        # Create chromosome and add it to the population
        chromosome = Chromosome.Chromosome(random_image_vector)
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

    #population = []
    # Initialize the population
    population = initialize_population(population_size)
    """image_path = "test_images/golden.jpg"  # Path to your image
    img = Image.open(image_path)
    chrom = Chromosome.Chromosome(img)
    population.append(chrom)

    image_path = "test_images/dog.jpg"  # Path to your image
    img = Image.open(image_path)
    chrom = Chromosome.Chromosome(img)
    population.append(chrom)"""

    # Evaluate the population
    elites = []
    evaluation.evaluate_population_with_sharing_function(population)
    for i in range(1, config.NUM_OF_GENERATIONS+1):
        print(f"Generation no: {i+1}\n")

        """# Get the elites if its not the first generation
        if i != 1:
            elites = selection.select_elites(population, elite_count)"""
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
        print("===========================================")


    """# Get the min and max fitness
    max_fitness, min_fitness = get_min_and_max_fitness(population)
    print(f"Max fitness: {max_fitness}")
    print(f"Min fitness: {min_fitness}\n------------------------------\n")

    for chrom in population:
        print(f"Chromosome fitness: {chrom.fitness}")
        print(f"Chromosome objective function: {chrom.obj_function}\n")"""


