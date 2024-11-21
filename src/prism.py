import sys
from contextlib import redirect_stdout
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

class PRISM:
    def __init__(self, N, K, reference_images,file_scheme):
        self.N = N  # Number of prompt generators
        self.K = K  # Training iterations per generator
        self.reference_images = reference_images
        self.best_generator = None
        self.best_score = float('-inf')
        self.generators = []  # List to store N prompt generators
        self.client = OpenAI()
        
        self.output_file = f"interpretable_feedback_and_prompts/prism_output_{file_scheme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def train_generator(self, generator, k):
        """Train a single generator for K iterations with feedback."""
        history = []
        best_local_score = float('-inf')
        best_local_prompt = None

        for i in range(k):
            try:
                # Sample reference image
                ref_image = random.choice(self.reference_images)
                print(f"\nIteration {i+1}: Using reference image {ref_image}")
                with open(self.output_file, 'a') as f:
                    f.write(f"\nIteration {i+1}: Using reference image {ref_image}")

                # Generate prompt based on history
                prompt = generator.generate(ref_image, reference_prompt=best_local_prompt)
                print(f"Generated prompt: {prompt[:100]}...")
                with open(self.output_file, 'a') as f:
                    f.write(f"\nGenerated prompt: {prompt}")
                
                # Generate image from prompt
                image_gen = ImageGenerator()
                generated_image = image_gen.generate(prompt)
                print("Generated new image")
                with open(self.output_file, 'a') as f:
                    f.write(f"\nGenerated new image")
                
                # Get similarity score
                scorer = SimilarityScorer()
                similarity_score = scorer.calculate_score(ref_image, generated_image)
                print(f"Similarity score: {similarity_score:.3f}")
                with open(self.output_file, 'a') as f:
                    f.write(f"\nSimilarity score: {similarity_score:.3f}")
                
                # Get feedback using GPT-4V
                feedback = self.get_feedback(ref_image, generated_image, similarity_score)
                print(f"Feedback received: {feedback[:100]}...")
                with open(self.output_file, 'a') as f:
                    f.write(f"\nFeedback received: {feedback}")
                
                # Update generator with feedback and score
                generator.update_with_feedback(feedback, similarity_score)
                
                if similarity_score > best_local_score:
                    best_local_score = similarity_score
                    best_local_prompt = prompt
                
                history.append({
                    'prompt': prompt,
                    'score': similarity_score,
                    'feedback': feedback
                })
                
                print(f"Completed iteration {i+1} with score {similarity_score:.3f}")
                with open(self.output_file, 'a') as f:
                    f.write(f"\nCompleted iteration {i+1} with score {similarity_score:.3f}")
                        
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                with open(self.output_file, 'a') as f:
                    f.write(f"Error in iteration {i+1}: {str(e)}")
                continue
            
        return best_local_prompt, best_local_score, history

    def get_feedback(self, original, generated, score):
        """Get feedback comparing original and generated images using GPT-4V."""
        try:
            prompt = f"""Compare these two images and provide specific feedback on:
            1. What aspects were captured well in the generated image
            2. What important elements were missed
            3. How to improve the prompt to get a more similar image
            Current similarity score: {score:.3f}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens = 1000,
                messages=[
                    {"role": "system", "content": "You are an expert image comparison assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": self._encode_image(original)}
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": self._encode_image(generated)}
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating feedback: {str(e)}"

    def _encode_image(self, image):
        """Helper method to encode image for API calls."""
        if isinstance(image, str):
            image = Image.open(image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def refine_prompts(self,file_scheme):
        """Main method to run the PRISM algorithm with multiple generators."""
        
        print("Initializing PRISM algorithm...")
        with open(self.output_file, 'a') as f:
            f.write("Initializing PRISM algorithm...")
        self.generators = [PromptGenerator() for _ in range(self.N)]
        generator_results = []

        for i, generator in enumerate(self.generators):
            print(f"\nTraining Generator {i+1}/{self.N}")
            with open(self.output_file, 'a') as f:
                f.write(f"\nTraining Generator {i+1}/{self.N}")
            best_prompt, best_score, history = self.train_generator(generator, self.K)
            
            generator_results.append({
                'generator': generator,
                'best_prompt': best_prompt,
                'best_score': best_score,
                'history': history
            })

        best_result = max(generator_results, key=lambda x: x['best_score'])
        self.best_generator = best_result['generator']
        self.best_score = best_result['best_score']

        print(f"\nBest Generator Score: {self.best_score:.3f}")
        with open(self.output_file, 'a') as f:
            f.write(f"\nBest Generator Score: {self.best_score:.3f}")
        print(f"Best Prompt: {best_result['best_prompt']}")
        with open(f"Best_Prompt_{file_scheme}", 'a') as f:
            f.write(f"\nBest Prompt: {best_result['best_prompt']}")
        
        return best_result['best_prompt']