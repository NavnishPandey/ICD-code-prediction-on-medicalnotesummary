
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

load_dotenv()

def setup_huggingface_auth():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face API token.")
    login(token)

class DiagnosisModel:
    def __init__(self, model_name="bvanaken/CORe-clinical-diagnosis-prediction"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
        self.label_map = self.model.config.id2label

    def predict(self, text: str, threshold=0.3):
        if not isinstance(text, str):
            raise ValueError("Expected input to be a string.")
        
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)
            predictions = (probs > threshold).nonzero(as_tuple=True)[1].tolist()
            return [self.label_map[i] for i in predictions]

class ICDPredictor:
    def __init__(self, model: DiagnosisModel):
        self.model = model

    def predict(self, summary: str):
        return self.model.predict(summary)
