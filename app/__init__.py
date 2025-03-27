from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from app.config.config import Config
from app.controllers.prediction_controller import PredictResource, ResultResource

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    CORS(app)
    api = Api(app)
    
    # Register routes
    api.add_resource(PredictResource, '/predict')
    api.add_resource(ResultResource, '/result/<string:image_id>')
    
    return app 