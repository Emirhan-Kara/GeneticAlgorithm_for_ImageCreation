import numpy as np
import config
import CNN
from PIL import Image, ImageDraw

class Chromosome:
    def __init__(self, gene_vector=None):
        # For debugging purposes, if the gene_vector is an actual image
        if isinstance(gene_vector, Image.Image):
            # Initialize with random triangles (we'll ignore the input image's content)
            # Just use image dimensions
            self.genes = self.create_random_gene_vector()
        elif gene_vector is None:
            # Create random gene vector if none provided
            self.genes = self.create_random_gene_vector()
        else:
            self.genes = np.array(gene_vector, dtype=np.float64)
    
        self.obj_function = self.calculate_obj_function()
        self.fitness = None

        # For roulette wheel selection
        self.p_i = None
        self.w_i = None
        # For rank selection
        self.r_i = None

    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def create_random_gene_vector(self):
        # Create a random gene vector for polygon-based representation
        # Each polygon has 3 points (x,y) and a color (r,g,b,a)
        # Format: [x1, y1, x2, y2, x3, y3, r, g, b, a]
        gene_vector = np.zeros((config.POLYGON_COUNT, config.POLYGON_PARAMS), dtype=np.float64)
        
        for i in range(config.POLYGON_COUNT):
            # Random points in range [0, 1]
            gene_vector[i, 0] = np.random.random()  # x1
            gene_vector[i, 1] = np.random.random()  # y1
            gene_vector[i, 2] = np.random.random()  # x2
            gene_vector[i, 3] = np.random.random()  # y2
            gene_vector[i, 4] = np.random.random()  # x3
            gene_vector[i, 5] = np.random.random()  # y3
            
            # Random RGBA color in range [0, 1]
            gene_vector[i, 6] = np.random.random()  # R
            gene_vector[i, 7] = np.random.random()  # G
            gene_vector[i, 8] = np.random.random()  # B
            gene_vector[i, 9] = np.random.random() * 0.5  # Alpha (semi-transparent)
            
        return gene_vector

    def reconstruct_image(self):
        # Create a blank white image
        img = Image.new('RGB', (config.CANVAS_SIZE, config.CANVAS_SIZE), color='white')
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw each triangle
        for i in range(config.POLYGON_COUNT):
            # Get triangle coordinates (scale from [0,1] to image dimensions)
            x1 = int(self.genes[i, 0] * config.CANVAS_SIZE)
            y1 = int(self.genes[i, 1] * config.CANVAS_SIZE)
            x2 = int(self.genes[i, 2] * config.CANVAS_SIZE)
            y2 = int(self.genes[i, 3] * config.CANVAS_SIZE)
            x3 = int(self.genes[i, 4] * config.CANVAS_SIZE)
            y3 = int(self.genes[i, 5] * config.CANVAS_SIZE)
            
            # Get RGBA color (scale RGB from [0,1] to [0,255])
            r = int(self.genes[i, 6] * 255)
            g = int(self.genes[i, 7] * 255)
            b = int(self.genes[i, 8] * 255)
            a = int(self.genes[i, 9] * 255)  # Alpha channel
            
            # Draw the triangle
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(r, g, b, a))
        
        return img
    
    def calculate_obj_function(self):
        # Get back the rendered image
        img = self.reconstruct_image()

        # Classify the image using the CNN model and return the sum of the scores for the target indices
        return CNN.classify_image(config.MODEL, img=img)
    
    
    # ============= Constrainted optimization with dynamic penalty =============
    # Constraint number 1: Probability >= 0.4
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