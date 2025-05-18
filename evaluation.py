import Chromosome
import config
import numpy as np

def dynmaic_penalty_function(chromosome, generation_counter, penalty_factor=0.5, alpha=1, beta=1, gamma=1):
        """
            This function calculates the dynamic penalty function for a chromosome

            T.C. -> O(n),       n=config.INEQUALITY_CONSTRAIN_NUM
            S.C. -> O(1)
        """
        # Calculate the penalty based on the fitness of the chromosome
        penalty_constant = pow((penalty_factor * generation_counter), alpha)
        bracket_g = 0.0
        for i in range(config.INEQUALITY_CONSTRAIN_NUM):
            bracket_g += pow(min(0, chromosome.inequality_constraint_function(i)), beta)

        penalty = penalty_constant * bracket_g

        return penalty

def evaluate_population_with_dynamic_penalty(population, generation_counter=1):
    """
        This function derives objective function for each chromosome in the population

        T.C. -> O(n),       n=config.POPULATION_SIZE
        S.C. -> O(1)
    """
    if config.IS_MAXIMIZATION:
        for chrom in population:
            chrom.fitness = chrom.obj_function - dynmaic_penalty_function(chrom, generation_counter)
    else:
        for chrom in population:
            chrom.fitness = chrom.obj_function + dynmaic_penalty_function(chrom, generation_counter)
    

# ===============================================================================================


def calculate_sigma_share(n = None, q = None, x_upper = None, x_lower = None):
    # Number of decision variables
    n = config.IMAGE_SIZE * config.IMAGE_SIZE * config.CHANNEL_NUM

    # For classification, each breed could represent a different optimum
    # For example fitness for dog class is calculated as the sum of 117 dog breeds
    # Each breed can be counted as a different local optimum
    q = len(config.TARGET_INDICES_DOG)

    # Upper and lower bounds for the decision variables
    x_upper = 1.0
    x_lower = 0.0

    sigma_share = (np.sqrt(n * ((x_upper - x_lower)**2))) / (q ** (1 / (2**n)))

    return sigma_share

def calculate_euclidean_distance(chromosome1, chromosome2):
    """
        This function calculates the Euclidean distance between two chromosomes

        T.C. -> O(n*n*m),   n=config.IMAGE_SIZE, m=config.CHANNEL_NUM
        S.C. -> O(n*n*m),   n=config.IMAGE_SIZE, m=config.CHANNEL_NUM
    """
    # Flatten the genes of both chromosomes and calculate the Euclidean distance
    return np.sqrt(np.sum(pow((chromosome1.genes.flatten() - chromosome2.genes.flatten()), 2)))

def Sh(d, sigma_share, alpha=1):
    """
        This function calculates the sharing function

        T.C. -> O(1)
        S.C. -> O(1)
    """
    # Calculate the sharing function
    if d <= sigma_share:
        return 1 - pow((d / sigma_share), alpha)
    return 0

def evaluate_population_with_sharing_function(population):
    sigma_share = calculate_sigma_share()
    distances = np.zeros((len(population), len(population)))
    sharing_function_values = np.ones((len(population), len(population)))
    niche_counts = np.zeros(len(population))

    # Calculate the pairwise euclidian distances and sharing function values
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[1]):
            # Calculate the Euclidean distance between two chromosomes
            distances[i][j] = calculate_euclidean_distance(population[i], population[j])
            distances[j][i] = distances[i][j]

            # Calculate the sharing function value
            sharing_function_values[i][j] = Sh(distances[i][j], sigma_share)
            sharing_function_values[j][i] = sharing_function_values[i][j]
    
    # Calculate the niche count for each chromosome
    for i in range(niche_counts.shape[0]):
        # Calculate the niche count for each chromosome
        niche_counts[i] = np.sum(sharing_function_values[i])

    # Calculate the fitness for each chromosome based on the niche count
    for i in range(len(population)):
        chrom = population[i]
        chrom.fitness = chrom.obj_function / niche_counts[i]
        

    