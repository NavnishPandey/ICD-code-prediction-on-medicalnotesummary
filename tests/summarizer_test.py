import pytest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.summarizer import OpenRouterClient, Summarizer


@patch.object(OpenRouterClient, "call", return_value="Patient reports no pain. Follow-up in 1 week.")
def test_summarizer_summary(mock_call):
    
    client = OpenRouterClient()
    summarizer = Summarizer(api_client=client)

    dialogue = "Doctor: How are you feeling today?\nPatient: I’m feeling better, no more pain."
    summary = summarizer.summarize(dialogue)

    assert "Follow-up" in summary or "no pain" in summary
    mock_call.assert_called_once()
    
@pytest.mark.parametrize("mock_summary", [
    "This is a summary.",
    "Patient has mild symptoms.",
    "Follow-up recommended in 1 week."
])
@patch("models.summarizer.requests.post")
def test_openrouterclient_call_various_summaries(mock_post, mock_summary):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": mock_summary
            }
        }]
    }
    mock_post.return_value = mock_response

    client = OpenRouterClient()
    result = client.call("Hello")

    assert result == mock_summary



@patch("models.summarizer.requests.post")
def test_openrouterclient_call_failure(mock_post):
    # Mock an exception on raise_for_status
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    mock_post.return_value = mock_response

    os.environ["OPENROUTER_API_KEY"] = "fake-key"

    client = OpenRouterClient()

    with pytest.raises(Exception) as excinfo:
        client.call("Test prompt")

    assert "API Error" in str(excinfo.value)

    del os.environ["OPENROUTER_API_KEY"]


def test_openrouterclient_missing_api_key(monkeypatch):
    # Ensure the environment variable is removed
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from models.summarizer import OpenRouterClient
    with pytest.raises(ValueError, match="Missing OpenRouter API Key"):
        OpenRouterClient()


