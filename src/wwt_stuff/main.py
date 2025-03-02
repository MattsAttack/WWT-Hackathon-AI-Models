import os
from pathlib import Path

from dotenv import load_dotenv
import fastapi
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel

from wwt_stuff.phishing_detector.bert_model.bert_run import Prediction

from .fact_checker import fact_check
from .json_streaming_response import JSONStreamingResponse
from .phishing_detector.bert_model import bert_run

p = Path(__file__).parent.resolve(strict=True)


load_dotenv()
MY_API = os.getenv("KEY")


app = fastapi.FastAPI()


client = genai.Client(api_key=MY_API)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/fact-checker")
async def fact_checker(website_url: str) -> fastapi.Response:
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


@app.get("/bg.png", include_in_schema=False)
async def bg() -> fastapi.Response:
    return Response(
        (p / "bg.png").read_bytes(),
        media_type="image/png",
    )


@app.get("/quack.mp3", include_in_schema=False)
async def quack() -> fastapi.Response:
    return Response(
        (p / "quack.mp3").read_bytes(),
        media_type="audio/mp3",
    )


class MessageResponse(BaseModel):
    response: str


@app.get("/chat")
async def chat(request: str) -> MessageResponse:
    if not request:
        raise HTTPException(status_code=400, detail="No message provided")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction="You are DUCKY, a friendly rubber duck cybersecurity and digital literacy assistant. You operate as a Chrome and Firefox extension. I have a phishing detector (automatic email scan), a chatbot (for information and questions), and a website fact checker (source verification). Use a playful, business-casual tone. Deliver concise responses under 75 words, and always append 'Quack!' to the end. No cursing. No blank lines. dont say Hey there very much. Dont be annoying.",
            ),
            contents=request,
        )

        if response.text is None:
            raise AssertionError

        return MessageResponse(response=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Wowzer") from e
