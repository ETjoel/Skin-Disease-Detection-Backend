import os
import uuid
from werkzeug.utils import secure_filename
from app.config.config import Config
import mimetypes

class FileService:
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        self._ensure_upload_folder()

    def _ensure_upload_folder(self):
        """Ensure the upload folder exists"""
        os.makedirs(self.upload_folder, exist_ok=True)

    def allowed_file(self, filename):
        """Check if the file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def get_mime_type(self, filename):
        """return file type"""
        return mimetypes.guess_type(filename)[0]

    def save_file(self, file):
        """Save the uploaded file and return the filepath"""
        if not file or file.filename == '':
            raise ValueError("No file provided")

        if not self.allowed_file(file.filename):
            raise ValueError("Invalid file type")

        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(self.upload_folder, unique_filename)
            
            file.save(filepath)
            return filepath, unique_filename
        except Exception as e:
            raise Exception(f"Error saving file: {e}") 