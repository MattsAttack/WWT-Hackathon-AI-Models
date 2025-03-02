import fastapi

from wwt_stuff.phishing_detector.bert_model.bert_run import Prediction

from .fact_checker import fact_check
from .json_streaming_response import JSONStreamingResponse
from .phishing_detector.bert_model import bert_run

app = fastapi.FastAPI()


@app.get("/fact-checker")
async def root(website_url: str) -> fastapi.Response:
    fact_checker = fact_check.AIFactChecker()
    response_stream = fact_checker.analyze(website_url)

    return JSONStreamingResponse(response_stream)


@app.get("/phishing")
async def phishing(subject: str, body: str) -> Prediction:
    return bert_run.predict_email(subject, body)
