import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback
import math
import cv2
from app.config.config import Config

class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained model, matching Django's load_model"""
        model_path = Config.MODEL_PATH
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            return model
        except OSError as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise Exception(f"Error loading model: {e}") from e

    # def analyze_model_structure(self, model):
    #     """Analyze model structure to understand expected input/output"""
    #     print("\n--- Model Structure Analysis ---")
    #     try:
    #         # Print overall model information
    #         print(f"Model type: {type(model)}")
    #         print(f"Number of layers: {len(model.layers)}")
            
    #         # Check the first few layers to understand input dimensions
    #         print("\nFirst layer:")
    #         first_layer = model.layers[0]
    #         print(f"  Type: {type(first_layer).__name__}")
    #         print(f"  Name: {first_layer.name}")
    #         if hasattr(first_layer, 'input_shape'):
    #             print(f"  Input shape: {first_layer.input_shape}")
    #         if hasattr(first_layer, 'output_shape'):
    #             print(f"  Output shape: {first_layer.output_shape}")
            
    #         # Check what comes before the dense layer
    #         for i, layer in enumerate(model.layers):
    #             if isinstance(layer, tf.keras.layers.Dense):
    #                 print(f"\nDense layer at position {i}:")
    #                 print(f"  Name: {layer.name}")
    #                 print(f"  Units: {layer.units}")
    #                 if hasattr(layer, 'input_shape'):
    #                     print(f"  Input shape: {layer.input_shape}")
    #                 if hasattr(layer, 'output_shape'):
    #                     print(f"  Output shape: {layer.output_shape}")
                    
    #                 # Look at the previous layer to understand what's feeding into the dense
    #                 if i > 0:
    #                     prev_layer = model.layers[i-1]
    #                     print(f"\nLayer before dense:")
    #                     print(f"  Type: {type(prev_layer).__name__}")
    #                     print(f"  Name: {prev_layer.name}")
    #                     if hasattr(prev_layer, 'output_shape'):
    #                         print(f"  Output shape: {prev_layer.output_shape}")
    #                 break
            
    #         # Look at model's expected input
    #         if hasattr(model, '_build_input_shape'):
    #             print(f"\nModel build input shape: {model._build_input_shape}")
            
    #         # Look at model's expected output
    #         print(f"\nOutput layer:")
    #         output_layer = model.layers[-1]
    #         print(f"  Type: {type(output_layer).__name__}")
    #         print(f"  Name: {output_layer.name}")
    #         if hasattr(output_layer, 'output_shape'):
    #             print(f"  Output shape: {output_layer.output_shape}")
    #         print(f"  Output units: {output_layer.units if hasattr(output_layer, 'units') else 'N/A'}")
            
    #         print("\n--- End Model Analysis ---\n")
    #     except Exception as e:
    #         print(f"Error during model analysis: {e}")
    #         traceback.print_exc()

    # def load_model(self):
    #     """Load the pre-trained model"""
    #     try:
    #         # Add compile=False to avoid optimizer state loading issues
    #         print(f"Loading model from {Config.MODEL_PATH}")
    #         self.model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
            
    #         # Analyze model structure to understand what's expected
    #         self.analyze_model_structure(self.model)
            
    #         # Print model summary for debugging
    #         print("Model architecture:")
    #         self.model.summary()
            
    #         # Get input shape safely
    #         input_shape = self.get_model_input_shape(self.model)
    #         print(f"Model input shape: {input_shape}")
            
    #         # Compile the model with basic settings after loading
    #         self.model.compile(
    #             optimizer='adam',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy']
    #         )
            
    #         print("Model loaded successfully")
    #     except Exception as e:
    #         print(f"Error loading model: {str(e)}")
    #         traceback.print_exc()
    #         self.model = None

    def image_to_model_input(self, image_path):
        """Preprocess the image for model input"""
        try:
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (28, 28))
            if len(resized_image.shape) == 2:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

            preprocessed_image = resized_image.astype('float32') / 255.0  
            return preprocessed_image
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict(self, image_path):
        """Make prediction on the image, matching Django's prediction logic"""
        classes = {
            4: 'Melanocytic Nevi (nv)',
            5: 'Melanocytic',
            6: 'Melanoma (mel)',
            2: 'Benign Keratosis-like Lesions (bkl)',
            1: 'Basal Cell Carcinoma (bcc)',
            5: 'Pyogenic Granulomas and Hemorrhage (vasc)',
            0: 'Actinic Keratoses and Intraepithelial Carcinomae (akiec)',
            3: 'Dermatofibroma (df)'
        }
        
        try:
            # Preprocess image
            processed_image = self.image_to_model_input(image_path)
            
            # Load model for this prediction
            model = self.load_model()
            
            # Add batch dimension and predict
            predictions = model.predict(np.expand_dims(processed_image, axis=0), verbose=0)
            print(f"Prediction output shape: {predictions.shape}")
            
            # Get predicted class
            predicted_class_index = np.argmax(predictions[0])
            
            predicted_class = classes[predicted_class_index]
            
            print(f"Predicted class: {predicted_class}")
            
            return {
                'disease': predicted_class,
                'predictions': predictions.tolist()
            }
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error making prediction: {str(e)}")