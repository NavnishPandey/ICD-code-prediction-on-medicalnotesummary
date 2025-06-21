from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from models.summarizer import Summarizer
from models.ICD_predictor import ICDPredictor

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class MedicalNotePipeline:
    def __init__(self):
        self.summarizer = Summarizer()
        self.predictor = ICDPredictor()

    async def index(self, request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    async def predict(self, request: Request, dialogue: str = Form(...)):
        summary = self.summarizer.summarize(dialogue)
        print(f"Generated Summary: {summary}")
        icd_code = self.predictor.predict(summary)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "dialogue": dialogue,
            "result": {
                "summary": summary,
                "icd_code": icd_code
            }
        })

pipeline_instance = MedicalNotePipeline()

# Bind class methods to FastAPI routes
app.get("/", response_class=HTMLResponse)(pipeline_instance.index)
app.post("/predict", response_class=HTMLResponse)(pipeline_instance.predict)




