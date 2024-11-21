from datetime import datetime
import os
import base64
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
import numpy as np
import random
import sys
from contextlib import redirect_stdout
import logging
from datetime import datetime

class ImageGenerator:
    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
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
            
            # Save the image to a file and return the file path
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"image_logs/generated_image_{current_time}.png"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(image_response.content)

            return file_path
            
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")