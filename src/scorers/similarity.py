import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine

class SimilarityScorer:
    def __init__(self):
        """Initialize CLIP model and processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def preprocess_image(self, image):
        """Convert image to CLIP input format."""
        if isinstance(image, str):
            image = Image.open(image)
        return self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        
    def extract_features(self, image):
        """Extract image features using CLIP."""
        with torch.no_grad():
            features = self.model.get_image_features(self.preprocess_image(image))
        return features.cpu().numpy()

    def calculate_score(self, original_image, generated_image):
        """
        Calculate similarity score between original and generated images using CLIP.
        
        Args:
            original_image: PIL Image or path to original image
            generated_image: PIL Image or path to generated image
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Extract features
            original_features = self.extract_features(original_image)
            generated_features = self.extract_features(generated_image)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(original_features.flatten(), generated_features.flatten())
            return float(similarity)
            
        except Exception as e:
            raise RuntimeError(f"Similarity calculation failed: {str(e)}")