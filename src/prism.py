import numpy as np
from .models.prompt_generator import PromptGenerator
from .scorers.similarity import SimilarityScorer
from .models.image_generator import ImageGenerator

class PRISM:
    def __init__(self, N, K, reference_images):
        self.N = N  # Number of streams
        self.K = K  # Number of iterations
        self.reference_images = reference_images  # Reference images {xi}M_i=1
        self.best_prompt = None
        self.best_score = float('-inf')

    def sample_reference_image(self):
        """Randomly sample a reference image."""
        return random.choice(self.reference_images)

    def generate_prompt(self, x):
        """Simulate prompt generation based on input x"""
        # x is the image
        prompt_generator = PromptGenerator()
        generated_prompt = prompt_generator.generate(x)
        return generated_prompt

    def generate_sampled_image(self, y):
        """Simulate image generation based on prompt y"""
        # Initialize image generator
        image_generator = ImageGenerator()
        # Generate image based on prompt
        generated_image = image_generator.generate(y)
        return f"Sampled image based on {y}"

    def calculate_in_iteration_score(self, x, sampled_x):
        """Calculate score based on original and sampled images)."""
        # Initialize similarity scorer
        scorer = SimilarityScorer()
        # Calculate similarity score between original and generated images
        score = scorer.calculate_score(x, sampled_x)
        return score  # Random score for demonstration

    def refine_prompts(self):
        """Main method to run the PRISM algorithm."""
        for n in range(self.N):  # Iterate over N streams
            chat_history = []  # Placeholder for chat history
            
            for k in range(self.K):  # Iterate over K iterations
                x_k_n = self.sample_reference_image()  # Sample a reference image
                y_k_n = self.generate_prompt(x_k_n)  # Generate a prompt
                
                sampled_x_k_n = self.generate_sampled_image(y_k_n)  # Sample an image based on the prompt
                
                score_prime = self.calculate_in_iteration_score(x_k_n, sampled_x_k_n)  # Calculate score

                # Update chat history and any other necessary parameters here (if needed)
                chat_history.append((x_k_n, y_k_n, sampled_x_k_n, score_prime))

            # Collecting best prompts from this stream based on scores
            best_prompts = sorted(chat_history, key=lambda x: x[3], reverse=True)[:self.N]  # Get top C best scores
            
            for yc in best_prompts:
                total_score = sum(self.calculate_in_iteration_score(xi, yc[1], self.generate_sampled_image(yc[1])) 
                                  for xi in self.reference_images)  # Re-evaluate with total score
                
                if total_score > self.best_score:
                    self.best_score = total_score
                    self.best_prompt = yc[1]  # Update best prompt

        return self.best_prompt

if __name__ == "__main__":
    reference_images = ["AAY.png", "ABPMJAY.png", "DDUGKY.png"]
    prism_algorithm = PRISM(N=5, K=10, reference_images=reference_images)
    best_prompt = prism_algorithm.refine_prompts()
    print(f"The best prompt is: {best_prompt}")