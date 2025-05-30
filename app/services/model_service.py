import tensorflow as tf
import numpy as np
from PIL import Image 
import os
import traceback
import cv2
from app.config.config import Config

# --- IMPORTANT: Re-calculate these values from your training notebook ---
# These should be the mean and standard deviation of the image pixel values
# *after* resizing to (100, 125) and *before* standardization, from your TRAINING dataset.
# In your notebook, this would be from the CNN section, for example:
# x_train_cnn_input = np.asarray(x_train_o['image'].tolist()) # After df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((125,100))))
# x_train_cnn_input = x_train_cnn_input.reshape(x_train_cnn_input.shape[0], *(100, 125, 3))
# print(f"Mean for CNN: {np.mean(x_train_cnn_input)}")
# print(f"Std for CNN: {np.std(x_train_cnn_input)}")
# Then copy those values here.
# These are placeholder values for 100x125 images:
X_TRAIN_MEAN = 112.75 # Placeholder - GET THIS FROM YOUR TRAINING DATA FOR 100x125 IMAGES
X_TRAIN_STD = 68.35  # Placeholder - GET THIS FROM YOUR TRAINING DATA FOR 100x125 IMAGES
# --- END IMPORTANT ---

class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained model"""
        # Ensure Config.MODEL_PATH points to "model.h5" if that's the CNN you want.
        model_path = Config.MODEL_PATH
        try:
            # It's good practice to set compile=False when loading for inference,
            # as optimizers might not be available or compatible across TensorFlow versions.
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully")
            
            # --- Added for Debugging ---
            print("\n--- Model Summary (for debugging input shape) ---")
            self.model.summary() # This will confirm the (None, 100, 125, 3) input shape.
            print("--- End Model Summary ---\n")
            # --- End Added ---
            
        except OSError as e:
            print(f"Error loading model from {model_path}: {e}")
            traceback.print_exc()
            raise Exception(f"Error loading model: {e}") from e

    def image_to_model_input(self, image_path):
        """Preprocess the image for CNN model input"""
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")

            # 1. Resize image to (100, 125) (height, width) as per the loaded CNN model's input expectations.
            # cv2.resize expects (width, height) tuple, so (125, 100)
            resized_image = cv2.resize(image, (125, 100)) 
            
            # 2. Convert BGR (OpenCV default) to RGB (PIL/Model training standard).
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # Ensure 3 channels in case it was loaded as grayscale (e.g., (H, W) or (H, W, 1))
            if len(resized_image.shape) == 2:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
            elif resized_image.shape[2] == 1: # If it's (height, width, 1) grayscale
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

            # 3. Apply the same standardization (mean/std normalization) used during training.
            preprocessed_image = resized_image.astype('float32')
            preprocessed_image = (preprocessed_image - X_TRAIN_MEAN) / X_TRAIN_STD
            
            return preprocessed_image # Shape will be (100, 125, 3)
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict(self, image_path):
        """Make prediction on the image"""
        classes = {
            0: 'Actinic Keratoses and Intraepithelial Carcinomae (akiec)',
            1: 'Basal Cell Carcinoma (bcc)',
            2: 'Benign Keratosis-like Lesions (bkl)',
            3: 'Dermatofibroma (df)',
            4: 'Melanocytic Nevi (nv)', 
            5: 'Pyogenic Granulomas and Hemorrhage (vasc)',
            6: 'Melanoma (mel)'
        }
        
        try:
            processed_image = self.image_to_model_input(image_path) # Returns shape (100, 125, 3)

            # Add a batch dimension: Transforms (100, 125, 3) to (1, 100, 125, 3)
            processed_image = np.expand_dims(processed_image, axis=0) 
            
            print(f"Input shape to model: {processed_image.shape}") 
            
            if self.model is None:
                raise Exception("Model not loaded. Ensure load_model() was successful.")
                
            predictions = self.model.predict(processed_image, verbose=0)
            # The last layer of your CNN model is softmax, so output is already probabilities.
            predictions = predictions[0] # Get the predictions for the single input image
            
            print(f"Prediction output (before argmax): {predictions}")
            
            # Log probabilities
            for idx, prob in enumerate(predictions):
                print(f"Class {classes.get(idx, 'Unknown')}: {prob:.6f}")
            
            # Validate predictions
            if np.any(predictions < 0) or np.any(predictions > 1):
                print(f"Warning: Predictions contain values outside [0, 1]: {predictions}")
            if not np.isclose(np.sum(predictions), 1.0, rtol=1e-5):
                print(f"Warning: Predictions sum to {np.sum(predictions):.6f}, not 1.0.")
            
            predicted_class_index = np.argmax(predictions)
            predicted_class = classes.get(predicted_class_index, "Unknown Class")
            print(f"Predicted class: {predicted_class}")
            
            return {
                'disease': predicted_class,
                'predictions': predictions.tolist()
            }
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error making prediction: {str(e)}")