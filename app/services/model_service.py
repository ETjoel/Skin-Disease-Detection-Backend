import tensorflow as tf
import numpy as np
from PIL import Image
from app.config.config import Config

class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(Config.MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, image_path):
        """Preprocess the image for model input"""
        try:
            img = Image.open(image_path)
            img = img.resize(Config.IMAGE_SIZE)
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {e}")

    def predict(self, image_path):
        """Make prediction on the image"""
        if self.model is None:
            raise Exception("Model not loaded")

        try:
            processed_image = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_image)
            
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_class = Config.CLASSES[predicted_class_index]
            
            return {
                'disease': predicted_class,
                'confidence': confidence
            }
        except Exception as e:
            raise Exception(f"Error making prediction: {e}") 