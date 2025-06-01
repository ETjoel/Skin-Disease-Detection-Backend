from flask import Flask, make_response
from flask_restful import Api
from flask_cors import CORS
from app.config.config import Config
from app.controllers.prediction_controller import PredictResource, ResultResource, Welcome
from app.controllers.gemini_controller import GeminiController
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    print(f"Upload directory set to: {Config.UPLOAD_FOLDER}")
    print(f"Model path set to: {Config.MODEL_PATH}")
    
    # Initialize extensions
    CORS(app)
    api = Api(app)

     # Global HEAD handler for health checks
    @app.route('/', methods=['HEAD'])
    def handle_head():
        return make_response("", 200)
    
    # Register routes
    api.add_resource(Welcome, '/')
    api.add_resource(PredictResource, '/predict')
    # api.add_resource(ResultResource, '/result/<string:image_id>')

    # Check if the image is skin skindisease
    api.add_resource(GeminiController, '/checkImage')
    
    return app 