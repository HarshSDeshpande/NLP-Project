import os
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO

class ImageGenerator:
    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
        os.environ["OPENAI_API_KEY"] = "your_api_key"
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.client = OpenAI()

    def generate(self, prompt: str) -> Image.Image:
        """
        Generate image using DALL-E 3 model.
        
        Args:
            prompt (str): Text prompt to generate image from
            
        Returns:
            PIL.Image: Generated image
        """
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            # Get image URL from response
            image_url = response.data[0].url
            
            # Download and convert to PIL Image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            return Image.open(BytesIO(image_response.content))
            
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")