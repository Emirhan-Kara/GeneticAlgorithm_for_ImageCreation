import numpy as np
import random
import config
import Chromosome

def create_parent_pairs(mating_pool):
    # Shufle the pool and pair the individuals
    random.shuffle(mating_pool)
    
    pairs = []
    for i in range(0, len(mating_pool)-1, 2):
        pairs.append((mating_pool[i], mating_pool[i+1]))
        
    return pairs

def sbx_crossover(parent1, parent2, eta=15):
    # Create empty gene arrays for children
    child1_genes = np.zeros_like(parent1.genes)
    child2_genes = np.zeros_like(parent2.genes)
    
    # Create the childen for each polygon and its parameters
    for i in range(parent1.genes.shape[0]):  # For each polygon
        for j in range(parent1.genes.shape[1]):  # For each parameter
            # Get parent gene values
            p1 = parent1.genes[i, j]
            p2 = parent2.genes[i, j]
            
            # If parents are identical, children are identical to parents
            if abs(p1 - p2) < 1e-10:
                child1_genes[i, j] = p1
                child2_genes[i, j] = p2
                continue
            
            # Ensure p1 <= p2
            if p1 > p2:
                p1, p2 = p2, p1
            
            # Calculate beta
            k = random.random()
            if k <= 0.5:
                beta = (2.0 * k) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - k))) ** (1.0 / (eta + 1.0))
            
            # Create children
            child1_genes[i, j] = 0.5 * ((p1 + p2) - (beta * (p2 - p1)))
            child2_genes[i, j] = 0.5 * ((p1 + p2) + (beta * (p2 - p1)))
            
            # Bound check
            child1_genes[i, j] = max(min(child1_genes[i, j], 1.0), 0.0)
            child2_genes[i, j] = max(min(child2_genes[i, j], 1.0), 0.0)
    
    # Create and return child chromosomes
    child1 = Chromosome.Chromosome(child1_genes)
    child2 = Chromosome.Chromosome(child2_genes)
    
    return child1, child2

def modified_random_mutation(chromosome, mutation_rate=config.MUTATION_RATE, delta=0.2):
    # Apply mutation to each gene with probability mutation_rate
    for i in range(chromosome.genes.shape[0]):  # For each polygon
        for j in range(chromosome.genes.shape[1]):  # For each parameter
            if random.random() < mutation_rate:
                r = random.random()
                y = chromosome.genes[i, j] + (delta * (r - 0.5))
                y = max(min(y, 1.0), 0.0)
                chromosome.genes[i, j] = y
                
                # Special handling for alpha channel - keep it semi-transparent
                if j == 9:  # Alpha parameter
                    chromosome.genes[i, j] = min(chromosome.genes[i, j], 0.7)  # Limit max alpha

def crossover_and_mutation(parent1, parent2, crossover_rate=config.CROSSOVER_RATE, mutation_rate=config.MUTATION_RATE):
    if random.random() < crossover_rate:
        child1, child2 = sbx_crossover(parent1, parent2)
    else:
        # Create deep copies to avoid modifying parents
        child1 = Chromosome.Chromosome(np.copy(parent1.genes))
        child2 = Chromosome.Chromosome(np.copy(parent2.genes))

    modified_random_mutation(child1, mutation_rate)
    modified_random_mutation(child2, mutation_rate)

    return child1, child2

def variation(mating_pool, crossover_rate=config.CROSSOVER_RATE, mutation_rate=config.MUTATION_RATE):
    Q_t = []
    
    parent_pairs = create_parent_pairs(mating_pool)

    for parent1, parent2 in parent_pairs:
        child1, child2 = crossover_and_mutation(parent1, parent2, crossover_rate, mutation_rate)
        Q_t.append(child1)
        Q_t.append(child2)

    return Q_t