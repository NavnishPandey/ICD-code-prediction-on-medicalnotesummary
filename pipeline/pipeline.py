from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from models.summarizer import Summarizer, OpenRouterClient
from models.ICD_predictor import ICDPredictor, DiagnosisModel, setup_huggingface_auth


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Dependency Injection ---
setup_huggingface_auth()

# Create required components
api_client = OpenRouterClient()
summarizer = Summarizer(api_client=api_client)

diagnosis_model = DiagnosisModel()  # 👈 Initialize this
predictor = ICDPredictor(model=diagnosis_model)  # 👈 Pass it in here

# Pipeline
class MedicalNotePipeline:
    def __init__(self, summarizer: Summarizer, predictor: ICDPredictor):
        self.summarizer = summarizer
        self.predictor = predictor

    async def index(self, request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    async def predict(self, request: Request, dialogue: str = Form(...)):
        summary = self.summarizer.summarize(dialogue)
        print(f"Generated Summary: {summary}")
        icd_codes = self.predictor.predict(summary)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "dialogue": dialogue,
            "result": {
                "summary": summary,
                "icd_code": icd_codes
            }
        })

pipeline = MedicalNotePipeline(summarizer=summarizer, predictor=predictor)

# --- Route Bindings ---
app.get("/", response_class=HTMLResponse)(pipeline.index)
app.post("/predict", response_class=HTMLResponse)(pipeline.predict)
