# summarizer.py
import os
import logging
from dotenv import load_dotenv
import requests
import json
load_dotenv()
logging.basicConfig(level=logging.INFO)

class Summarizer:
    def __init__(self, model="mistralai/ministral-3b"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        if not self.api_key:
            raise ValueError("Missing OpenRouter API Key")

    def summarize(self, dialogue: str) -> str:
        print(f"Summarizing dialogue: {dialogue}")
        prompt = (
            "You are a helpful assistant. Summarize the following medical conversation "
            "between a doctor and a patient in a concise and clear note.\n\n"
            f"{dialogue}\n\nSummary:"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    

        response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers=headers,
  data=json.dumps({
    "model": self.model, 
    "messages": [
      {
        "role": "user",
        "content": prompt
      }
    ]
  })
)
        
        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        return generated_text
