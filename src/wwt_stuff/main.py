from pathlib import Path

import fastapi
from fastapi.responses import HTMLResponse, Response

from wwt_stuff.phishing_detector.bert_model.bert_run import Prediction

from .fact_checker import fact_check
from .json_streaming_response import JSONStreamingResponse
from .phishing_detector.bert_model import bert_run

p = Path(__file__).parent.resolve(strict=True)

app = fastapi.FastAPI()


@app.get("/fact-checker")
async def root(website_url: str) -> fastapi.Response:
    fact_checker = fact_check.AIFactChecker()
    response_stream = fact_checker.analyze(website_url)

    return JSONStreamingResponse(response_stream)


@app.get("/phishing")
async def phishing(subject: str, body: str) -> Prediction:
    return bert_run.predict_email(subject, body)


@app.get("/")
async def home() -> fastapi.Response:
    return HTMLResponse((p / "index.html").read_text(encoding="utf-8"))


@app.get("/ducky.svg", include_in_schema=False)
async def favicon() -> fastapi.Response:
    return Response(
        (p / "ducky.svg").read_text(encoding="utf-8"),
        media_type="image/svg+xml",
    )
