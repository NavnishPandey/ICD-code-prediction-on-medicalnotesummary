import ray
from ray import serve
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pipeline.pipeline import MedicalNotePipeline
import uvicorn
from pipeline.pipeline import app

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token)

if __name__ == "__main__":
    print("🚀 App running at http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
