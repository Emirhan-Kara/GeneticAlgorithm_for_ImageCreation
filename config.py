import numpy as np

# Image settings
CANVAS_SIZE = 224  # Output image size
POLYGON_COUNT = 50  # Number of triangles per individual
POLYGON_PARAMS = 10  # Parameters per triangle: 6 for coordinates, 4 for RGBA
SAVE_EVERY_N_GENERATIONS = 100
MODEL = None

# Classification settings
TARGET_INDICES_DOG = np.arange(151, 268)

# Genetic algorithm settings
IS_MAXIMIZATION = True
OBJECTIVE_NUM = 1
INEQUALITY_CONSTRAIN_NUM = 1
EQUALITY_CONSTRAIN_NUM = 0
CONSTRAIN_THRESHOLD_1 = 0.4

NUM_OF_GENERATIONS = 10000
POPULATION_SIZE = 100
ELITE_COUNT = (POPULATION_SIZE * 5) // 100
MATING_POOL_SIZE = POPULATION_SIZE
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 2