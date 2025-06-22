from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pipeline.pipeline import app

client = TestClient(app)

dialogue_text = """
Doctor: How are you feeling today?
Patient: I've been having a mild cough and slight fever since yesterday.
Doctor: Any chest pain or difficulty breathing?
Patient: No, just the cough and fever.
Doctor: Alright, I'll note that down and prescribe medication accordingly.
"""

@patch("pipeline.pipeline.Summarizer.summarize")
@patch("pipeline.pipeline.ICDPredictor.predict")
def test_predict_route(mock_predict_predict, mock_summarize):
    # Setup return values
    mock_summarize.return_value = "mocked summary"
    mock_predict_predict.return_value = ["CODE1", "CODE2"]

    response = client.post("/predict", data={"dialogue": dialogue_text})

    assert response.status_code == 200

    # Check that these methods were called exactly once
    mock_summarize.assert_called_once()
    mock_predict_predict.assert_called_once()

    # Optional: loose check in response text
    assert "mocked summary" in response.text or "CODE1" in response.text or "CODE2" in response.text

def test_predict_route_missing_form_data():
    response = client.post("/predict", data={})
    assert response.status_code == 422
