from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token)

class ICDPredictor:
    def __init__(self, model_name="bvanaken/CORe-clinical-diagnosis-prediction"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

    def predict(self, summary: str):
        print(f"Type of summary: {type(summary)}")  # Debug print
        if not isinstance(summary, str):
            raise ValueError("Expected input summary to be a string.")
        
        tokenized_input = self.tokenizer(summary, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**tokenized_input)
        predictions = torch.sigmoid(output.logits)
        predicted_labels = [
            self.model.config.id2label[_id]
            for _id in (predictions > 0.3).nonzero()[:, 1].tolist()
        ]
        return predicted_labels
