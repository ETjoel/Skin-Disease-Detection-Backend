from flask import request
from flask_restful import Resource
from app.services.model_service import ModelService
from app.services.file_service import FileService

class PredictResource(Resource):
    def __init__(self):
        self.model_service = ModelService()
        self.file_service = FileService()

    def post(self):
        """Handle image upload and prediction"""
        try:
            if 'file' not in request.files:
                return {'error': 'No file provided'}, 400
            
            file = request.files['file']
            filepath, unique_filename = self.file_service.save_file(file)
            
            # Get prediction from model
            prediction = self.model_service.predict(filepath)
            
            # Combine results
            result = {
                'image_id': unique_filename,
                **prediction
            }
            
            return result, 200
            
        except ValueError as e:
            return {'error': str(e)}, 400
        except Exception as e:
            return {'error': str(e)}, 500

class ResultResource(Resource):
    def get(self, image_id):
        """Retrieve prediction result for a specific image"""
        # In a real application, you would fetch this from a database
        # For this example, we'll return a mock response
        return {
            'image_id': image_id,
            'disease': 'melanoma',
            'confidence': 0.95
        }, 200 