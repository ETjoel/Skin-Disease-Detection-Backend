import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback
import math
from app.config.config import Config

class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def get_model_input_shape(self, model):
        """Extract input shape from model safely"""
        try:
            # Try to get input shape from model config
            if hasattr(model, 'input_shape'):
                return model.input_shape
            
            # Try to get from input layer
            if hasattr(model, 'inputs') and model.inputs:
                return model.inputs[0].shape
            
            # Try to get from the first layer's input spec
            if hasattr(model, 'layers') and model.layers and hasattr(model.layers[0], 'input_spec'):
                if model.layers[0].input_spec and hasattr(model.layers[0].input_spec[0], 'shape'):
                    return model.layers[0].input_spec[0].shape
            
            # Last resort: assume it's a CNN with standard input 
            return (None, 28, 28, 3)  # Default to what we know works
        except Exception as e:
            print(f"Error getting model input shape: {e}")
            traceback.print_exc()
            return (None, 28, 28, 3)  # Fallback

    def analyze_model_structure(self, model):
        """Analyze model structure to understand expected input/output"""
        print("\n--- Model Structure Analysis ---")
        try:
            # Print overall model information
            print(f"Model type: {type(model)}")
            print(f"Number of layers: {len(model.layers)}")
            
            # Check the first few layers to understand input dimensions
            print("\nFirst layer:")
            first_layer = model.layers[0]
            print(f"  Type: {type(first_layer).__name__}")
            print(f"  Name: {first_layer.name}")
            if hasattr(first_layer, 'input_shape'):
                print(f"  Input shape: {first_layer.input_shape}")
            if hasattr(first_layer, 'output_shape'):
                print(f"  Output shape: {first_layer.output_shape}")
            
            # Check what comes before the dense layer
            for i, layer in enumerate(model.layers):
                if isinstance(layer, tf.keras.layers.Dense):
                    print(f"\nDense layer at position {i}:")
                    print(f"  Name: {layer.name}")
                    print(f"  Units: {layer.units}")
                    if hasattr(layer, 'input_shape'):
                        print(f"  Input shape: {layer.input_shape}")
                    if hasattr(layer, 'output_shape'):
                        print(f"  Output shape: {layer.output_shape}")
                    
                    # Look at the previous layer to understand what's feeding into the dense
                    if i > 0:
                        prev_layer = model.layers[i-1]
                        print(f"\nLayer before dense:")
                        print(f"  Type: {type(prev_layer).__name__}")
                        print(f"  Name: {prev_layer.name}")
                        if hasattr(prev_layer, 'output_shape'):
                            print(f"  Output shape: {prev_layer.output_shape}")
                    break
            
            # Look at model's expected input
            if hasattr(model, '_build_input_shape'):
                print(f"\nModel build input shape: {model._build_input_shape}")
            
            # Look at model's expected output
            print(f"\nOutput layer:")
            output_layer = model.layers[-1]
            print(f"  Type: {type(output_layer).__name__}")
            print(f"  Name: {output_layer.name}")
            if hasattr(output_layer, 'output_shape'):
                print(f"  Output shape: {output_layer.output_shape}")
            print(f"  Output units: {output_layer.units if hasattr(output_layer, 'units') else 'N/A'}")
            
            print("\n--- End Model Analysis ---\n")
        except Exception as e:
            print(f"Error during model analysis: {e}")
            traceback.print_exc()

    def load_model(self):
        """Load the pre-trained model"""
        try:
            # Add compile=False to avoid optimizer state loading issues
            print(f"Loading model from {Config.MODEL_PATH}")
            self.model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
            
            # Analyze model structure to understand what's expected
            self.analyze_model_structure(self.model)
            
            # Print model summary for debugging
            print("Model architecture:")
            self.model.summary()
            
            # Get input shape safely
            input_shape = self.get_model_input_shape(self.model)
            print(f"Model input shape: {input_shape}")
            
            # Compile the model with basic settings after loading
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            self.model = None

    def preprocess_image(self, image_path):
        """Preprocess the image for model input"""
        try:
            if not os.path.exists(image_path):
                raise Exception(f"Image file not found: {image_path}")
                
            img = Image.open(image_path)
            
            # The model expects input shape (None, 28, 28, 3)
            target_size = (28, 28)
            print(f"Using specific dimensions for this model: {target_size}")
            
            # Resize the image
            img = img.resize(target_size)
            print(f"Resized image to: {target_size}")
            
            # Convert to numpy array
            img_array = np.array(img)
            print(f"Image array shape after resize: {img_array.shape}")
            
            # Handle both RGB and RGBA images
            if len(img_array.shape) == 3 and img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:,:,:3]  # Convert to RGB
                print("Converted RGBA to RGB")
            
            # Ensure image has 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack((img_array,) * 3, axis=-1)
                print("Converted grayscale to RGB")
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            # Add batch dimension to create 4D tensor (batch, height, width, channels)
            # Model expects shape (None, 28, 28, 3)
            processed_image = np.expand_dims(img_array, axis=0)
            print(f"Final processed image shape: {processed_image.shape}")
            
            return processed_image
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict(self, image_path):
        """Make prediction on the image"""
        if self.model is None:
            try:
                print("Model not loaded, attempting to reload...")
                self.load_model()
                if self.model is None:
                    raise Exception("Model could not be loaded")
            except Exception as e:
                print(f"Failed to reload model: {str(e)}")
                traceback.print_exc()
                raise Exception("Model not loaded")

        try:
            processed_image = self.preprocess_image(image_path)
            print(f"Input shape for prediction: {processed_image.shape}")
            
            # Check the expected input shape for the model
            if hasattr(self.model, 'input_shape'):
                print(f"Model expects input shape: {self.model.input_shape}")
            
            predictions = self.model.predict(processed_image, verbose=0)
            print(f"Prediction output shape: {predictions.shape}")
            
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_class = Config.CLASSES[predicted_class_index]
            
            print(f"Predicted class: {predicted_class}, confidence: {confidence}")
            
            return {
                'disease': predicted_class,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error making prediction: {str(e)}") 