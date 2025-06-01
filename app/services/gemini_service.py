from google import genai
from google.genai import types
from app.config.config import Config
import mimetypes


class GeminiService:
    def __init__(self):
        self.client = None
        self.load_client()

    def load_client(self):

        gemini_key = Config.GEMINI_KEY
        try:
            self.client = genai.Client(api_key=gemini_key)

        except Exception as e:
            print(f"Error loading client {e} ")
            raise e
    
    def check_skin_disease(self,image, mime_type) -> str:
        system_instruction = "You are an AI assistant helping to filter images. Your task is to determine if an image is relevant for skin disease analysis. Only respond with 'SKIN_RELATED' or 'NON_SKIN_RELATED' followed by a brief justification if NON_SKIN_RELATED."
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    types.Part.from_bytes(
                        data=image,
                        mime_type=mime_type,
                    ),
                    system_instruction
                ],
            )
            
            # Assuming response returns a string like 'SKIN_RELATED' or 'NON_SKIN_RELATED: reason'
            return response.text.strip()
            
        except Exception as e:
            print(f"Unable to generate content with Error: {e}")
            raise e
