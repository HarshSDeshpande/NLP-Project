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

class PromptGenerator:
    def __init__(self):
        """Initialize OpenAI client with learning capabilities."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.client = OpenAI()
        self.feedback_history = []  # Store (prompt, feedback, score) tuples
        self.conversation_history = []
        self.successful_patterns = []  # Track what works well
        self.improvement_areas = []  # Track what needs improvement

    def encode_image(self, image):
        """Convert image to base64 string."""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Image encoding failed: {str(e)}")

    def update_with_feedback(self, feedback, score):
        """
        Learn from feedback and score to improve future prompts.

        Args:
            feedback (str): Feedback from image comparison
            score (float): Similarity score between 0 and 1
        """
        try:
            analysis = self.analyze_feedback(feedback, score)
            self.feedback_history.append({
                'feedback': feedback,
                'score': score,
                'analysis': analysis
            })
            
            # Classify feedback based on score
            if score > 0.7:  # High similarity
                self.successful_patterns.append(analysis)
            else:  # Needs improvement
                self.improvement_areas.append(analysis)
                
        except Exception as e:
            print(f"Failed to update with feedback: {str(e)}")

    def analyze_feedback(self, feedback, score):
        """Extract learning points from feedback with score context."""
        try:
            analysis_prompt = f"""Based on this feedback and similarity score ({score:.3f}), identify:
            1. What worked well in the prompt (if score > 0.7)
            2. What needs improvement (if score < 0.7)
            3. Specific patterns to replicate or avoid"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are analyzing prompt generation feedback."},
                    {"role": "user", "content": f"Score: {score}\nFeedback: {feedback}"}
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Feedback analysis failed: {str(e)}")
            return None

    def generate(self, image, reference_prompt=None) -> str:
        """Generate improved prompt using learned patterns."""
        try:
            base64_image = self.encode_image(image)
            
            # Create learning-focused system prompt
            system_prompt = """You are an expert prompt engineer who learns and improves.
            Previously successful patterns included: {}\n
            Areas for improvement were: {}\n
            Generate a detailed image description incorporating these learnings.
            ENSURE THAT EVERYTHING GENERATED IS IN COMPLIANCE WUTH OPENAI'S CONTENT GENERATION POLICY""".format(
                '; '.join(self.successful_patterns[-3:]) if self.successful_patterns else "None yet",
                '; '.join(self.improvement_areas[-3:]) if self.improvement_areas else "None yet"
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add relevant feedback history
            if self.feedback_history:
                recent_learnings = [f"Score {f['score']}: {f['analysis']}" 
                                  for f in self.feedback_history[-3:]]
                messages.append({
                    "role": "user",
                    "content": "Recent feedback learnings:\n" + "\n".join(recent_learnings)
                })
            
            # Add image and current request
            user_content = [
                {"type": "text", "text": "Generate an improved image description based on our learnings."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
            
            if reference_prompt:
                user_content[0]["text"] += f"\nReference: {reference_prompt}"
            
            messages.append({"role": "user", "content": user_content})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            prompt = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": prompt
            })
            
            return prompt
            
        except Exception as e:
            raise RuntimeError(f"Prompt generation failed: {str(e)}")