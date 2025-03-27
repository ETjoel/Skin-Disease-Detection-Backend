# Skin Disease Detection API

This Flask application provides an API for skin disease detection using a pre-trained TensorFlow model. The application follows a modular architecture using the MVC pattern.

## Project Structure

```
app/
├── config/
│   └── config.py         # Configuration settings
├── controllers/
│   └── prediction_controller.py  # API endpoints
├── models/               # Data models (to be implemented)
├── services/
│   ├── model_service.py  # Model-related operations
│   └── file_service.py   # File handling operations
└── utils/               # Utility functions
run.py                   # Application entry point
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your pre-trained model file (`model.h5`) in the root directory of the project.

## Running the Application

Start the Flask server:
```bash
python run.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /predict
Upload an image for disease detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - file: Image file (PNG, JPG, JPEG)

**Response:**
```json
{
    "image_id": "unique_filename",
    "disease": "predicted_disease",
    "confidence": 0.95
}
```

### GET /result/<image_id>
Retrieve the prediction result for a previously uploaded image.

**Request:**
- Method: GET
- URL: `/result/<image_id>`

**Response:**
```json
{
    "image_id": "image_id",
    "disease": "predicted_disease",
    "confidence": 0.95
}
```

## Error Handling

The API returns appropriate error messages with HTTP status codes:
- 400: Bad Request (invalid file, no file provided)
- 500: Internal Server Error (processing errors)

## Notes

- The application expects images to be preprocessed to match the model's input requirements (224x224 pixels)
- Supported image formats: PNG, JPG, JPEG
- CORS is enabled for frontend integration 