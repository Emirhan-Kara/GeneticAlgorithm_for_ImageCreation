import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.applications import EfficientNetB0
from keras.api.preprocessing import image
from keras.api.applications.efficientnet import preprocess_input, decode_predictions
import os
import config

# Constants
IMAGE_PATH = "test_images/random_squares.png"  # Path to your image

def load_model():
    """Load the EfficientNet model"""
    print("Loading EfficientNet model...")
    model = EfficientNetB0(weights='imagenet', include_top=True)
    print("Model loaded successfully!")
    return model

def sum_class_scores(preds, class_indices):
    """Sum the prediction scores for specified class indices"""
    preds = preds[0] if len(preds.shape) > 1 else preds
    return float(sum(preds[idx] for idx in class_indices))

def classify_image(model=config.MODEL, img=None, img_path=None, class_indices=config.TARGET_INDICES_DOG):
    """Classify an image and return sum of confidences for specified class indices"""
    if img is None and img_path is not None:
        img = image.load_img(img_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    elif img is None:
        raise ValueError("Either img or img_path must be provided")
    
    img_array = image.img_to_array(img)
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return sum_class_scores(preds, class_indices)

def display_results(model, img_path, class_indices):
    """Display an image and its classification results for specified class indices"""
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} not found")
        return
    img = plt.imread(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')

    total_score = classify_image(model, img_path, class_indices)
    plt.title(f"Dog probability: {total_score:.4f}")
    print(f"Dog probability: {total_score:.4f}")
    plt.show()

# Main execution
if __name__ == "__main__":
    model = load_model()
    display_results(model, IMAGE_PATH, class_indices=TARGET_INDICES)
    total_score = classify_image(model, IMAGE_PATH, class_indices=TARGET_INDICES)
    print(f"\nDog probability: {total_score:.4f}")
