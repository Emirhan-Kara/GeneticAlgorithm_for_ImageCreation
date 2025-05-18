import numpy as np
import config
import CNN
from PIL import Image

class Chromosome:
    def __init__(self, gene_vector):
        # For debugging purposes, if the gene_vector is an actual image
        if isinstance(gene_vector, Image.Image):
            # Resize the image and convert to RGB
            gene_vector = gene_vector.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
            gene_vector = gene_vector.convert('RGB')

            # Convert to numpy array and normalize to [0,1]
            gene_vector = np.array(gene_vector).astype(np.float64) / 255.0
        else:
            gene_vector = np.array(gene_vector, dtype=np.float64)
    
        self.genes = gene_vector
        self.obj_function = self.calculate_obj_function()
        self.fitness = 0.0

    def reconstruct_image(self):
        # Convert from float64 [0-1] to uint8 [0-255]
        img_array = (self.genes * 255).astype(np.uint8)
        
        # Create image
        return Image.fromarray(img_array)
    
    def calculate_obj_function(self):
        # Get back the original image
        img = self.reconstruct_image()

        # Classify the image using the CNN model and return the sum of the scores for the target indices
        return CNN.classify_image(config.MODEL, img=img)
    
    
    # ============= Constrainted optimization with dynamic penalty =============
    # Constraint number 1: Probabilirt >= 0.4
    # To make it >=, function g becomes g(x) = f(x) - 0.4 >= 0.0
    def inequality_constraint_function(self, constraint_idx):
        """
            This function calculates the first inequality constraint function for a chromosome

            T.C. -> O(1)
            S.C. -> O(1)
        """
        # Calculate the constraint function
        if constraint_idx == 0:
            # constraint function
            return self.obj_function - config.CONSTRAIN_THRESHOLD_1
        
        raise ValueError(f"Invalid constraint index: {constraint_idx}")

    # ==========================================================================