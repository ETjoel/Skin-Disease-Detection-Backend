import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model configuration
    MODEL_PATH = 'model.h5'
    IMAGE_SIZE = (224, 224)
    
    # Disease classes
    CLASSES = ['acne', 'eczema', 'melanoma', 'normal', 'psoriasis'] 