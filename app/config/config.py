import os

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model configuration
    MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
    IMAGE_SIZE = (224, 224)
    
    # Specific model configuration based on the error message
    # Input shape for the dense layer is (2304)
    MODEL_INPUT_DIMS = (27, 28, 3)  # 27*28*3 = 2268 (close to 2304)
    # Alternative (based on error): sqrt(2304/3) = 27.71... ~ (28, 28, 3)
    
    # Disease classes
    CLASSES = ['acne', 'eczema', 'melanoma', 'normal', 'psoriasis'] 