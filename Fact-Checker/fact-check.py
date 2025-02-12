from collections.abc import AsyncGenerator
import re
from typing import Never
from urllib.parse import quote

from bs4 import BeautifulSoup
import httpx
from pydantic import BaseModel
from rich.progress import Progress
from rich.table import Table
from sentence_transformers import SentenceTransformer, util
from torch.types import Number


class Source(BaseModel):
    title: str
    snippet: str
    link: str


class Result(BaseModel):
    claim: str
    evidence: Source | None
    confidence: Number


_MIN_SENTENCE_LENGTH = 20


class AIFactChecker:
    def __init__(self) -> None:
        """Load models"""
        self.evidence_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def extract_content(self, url: str) -> str:
        """Scrape and clean website content."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract core content (e.g., article text)
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])

    def detect_claims(self, content: str) -> list[str]:
        """Identify claims within the content."""
        return [
            sentence.strip()
            for sentence in (re.compile(r"[.!\n]").split(content))
            # Basic length filter
            if len(sentence.strip()) > _MIN_SENTENCE_LENGTH
            and not re.match(r"(Copyright S&P|provided by|© 20)", sentence)
        ]

    async def retrieve_evidence(
        self,
        claim: str,
    ) -> list[Source]:
        """Search for evidence related to a claim."""
        query_url = f"https://www.googleapis.com/customsearch/v1?q={quote(claim)}&key=API_KEY&cx=CX_ID"
        async with httpx.AsyncClient() as client:
            response = (await client.get(query_url)).json()
        return [Source.model_validate(source) for source in response.get("items", [])]

    def verify_claim(
        self,
        claim: str,
        evidence: list[Source],
    ) -> tuple[Source | None, Number]:
        """Compare claim with evidence using semantic similarity."""
        evidence_texts = [e.snippet for e in evidence]
        similarities: list[Number] = []

        for evidence_text in evidence_texts:
            similarity = util.pytorch_cos_sim(
                self.evidence_model.encode(claim),
                self.evidence_model.encode(evidence_text),
            )
            similarities.append(similarity.item())

        # Take the evidence with the highest similarity
        if similarities:
            best_match_index = similarities.index(max(similarities))
            return evidence[best_match_index], max(similarities)

        return None, 0

    async def analyze(
        self, url: str, progress: Progress
    ) -> AsyncGenerator[Result, Never]:
        """End-to-end analysis: Extract, detect claims, and fact-check."""
        content = await self.extract_content(url)
        claims = self.detect_claims(content)

        for claim in progress.track(claims):
            evidence: list[Source] = await self.retrieve_evidence(claim)
            best_evidence, confidence = self.verify_claim(claim, evidence)
            yield (Result(claim=claim, evidence=best_evidence, confidence=confidence))


async def main() -> None:
    website_url = "https://www.cnn.com/2025/02/09/business/trump-tariffs-steel-aluminum/index.html"

    with Progress() as progress:
        progress.console.print(f"Beginning analysis of {website_url}")

        fact_checker = AIFactChecker()
        results = fact_checker.analyze(website_url, progress)

        table = Table(title="Fact Check Results", show_lines=True)
        table.add_column("Claim")
        table.add_column("Evidence")
        table.add_column("Confidence", highlight=True)

        async for result in results:
            table.add_row(
                result.claim,
                str(result.evidence),
                f"{result.confidence:.2f}",
            )

        progress.console.print(table)


if __name__ == "__main__":
    import trio

    trio.run(main)
