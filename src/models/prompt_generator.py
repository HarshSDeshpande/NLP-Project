import os
from openai import OpenAI
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class PromptGenerator:
    def __init__(self):
        """Initialize OpenAI client and CLIP model for image understanding."""
        os.environ["OPENAI_API_KEY"] = "your_api_key"
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.client = OpenAI()
        
        # Initialize CLIP for image understanding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def extract_image_features(self, image):
        """Extract features from image using CLIP."""
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().cpu().numpy()

    def generate(self, image) -> str:
        """
        Generate image prompt from input image using CLIP and GPT-4.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            str: Generated image prompt
        """
        try:
            # Extract image features
            features = self.extract_image_features(image)
            
            # Create prompt for GPT-4
            system_prompt = """You are a helpful assistant who needs to describe an image in detail for 
            generating a similar image. Focus on visual elements, composition, style, colors, and mood. 
            Do not include text elements. Ensure the description is compliant with OpenAI's policies."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a detailed description for recreating this image."}
                ],
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.7
            )
            
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Prompt generation failed: {str(e)}")