

import os
import logging
import json
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- API Client ---
class OpenRouterClient:
    def __init__(self, model="mistralai/ministral-3b"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model

        if not self.api_key:
            raise ValueError("Missing OpenRouter API Key")

    def call(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]


class Summarizer:
    def __init__(self, api_client):
        self.api_client = api_client

    def summarize(self, dialogue: str) -> str:
        logging.info("Summarizing dialogue...")
        prompt = (
            "You are a helpful assistant. Summarize the following medical conversation "
            "between a doctor and a patient in a concise and clear note.\n\n"
            f"{dialogue}\n\nSummary:"
        )
        return self.api_client.call(prompt)
