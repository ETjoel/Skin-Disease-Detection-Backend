from flask import request
from flask_restful import Resource
from app.services.gemini_service import GeminiService
from app.services.file_service import FileService

class GeminiController(Resource):
    def __init__(self):
        self.gemini_service = GeminiService()
        self.file_service = FileService()

    def post(self):
        """Handle image upload and prediction"""

        filepath = None
        try:
            if 'file' not in request.files:
                return {'error': 'No file provided'}, 400
            
            file = request.files['file']
            if not file or file.filename == '':
                return {'error': 'Empty file provided'}, 400

            # Validate file type before saving
            if not self.file_service.allowed_file(file.filename):
                return {'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}, 400
            
            mime_type = self.file_service.get_mime_type(file.filename)

             # Validate MIME type
            valid_mime_types = {'image/jpeg', 'image/png', 'image/jpg'}
            if not mime_type or mime_type not in valid_mime_types:
                return {'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}, 400
            
            # Read file stream
            image_data = file.stream.read()
            if not image_data:
                return {'error': 'File is empty or corrupted', 'success': False}, 400

            response = self.gemini_service.check_skin_disease(image_data, mime_type)

            result = {
                'success': True,
                'is_skin_disease': True if response == 'SKIN_RELATED' else False
            }
            return result, 200
        except Exception as e:
            print(f"Unable to generate content with Error: {e}")
            return {'error': str(e), 'success': False}, 500

            