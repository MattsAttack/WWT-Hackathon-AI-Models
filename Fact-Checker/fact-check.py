from bs4 import BeautifulSoup
import httpx
from pydantic import BaseModel
import rich
from sentence_transformers import SentenceTransformer, util
from torch.types import Number
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


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
        self.claim_detection_model = pipeline("ner", model="dslim/bert-base-NER")
        self.evidence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.verification_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/t5-small-ssm-nq",
        )
        self.verification_tokenizer = AutoTokenizer.from_pretrained(
            "google/t5-small-ssm-nq",
        )

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
            for sentence in content.split(".")
            # Basic length filter
            if len(sentence.strip()) > _MIN_SENTENCE_LENGTH
        ]

    async def retrieve_evidence(
        self,
        claim: str,
    ) -> list[Source]:
        """Search for evidence related to a claim."""
        query_url = (
            f"https://www.googleapis.com/customsearch/v1?q={claim}&key=API_KEY&cx=CX_ID"
        )
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

    async def analyze(self, url: str) -> list[Result]:
        """End-to-end analysis: Extract, detect claims, and fact-check."""
        content = await self.extract_content(url)
        claims = self.detect_claims(content)
        results: list[Result] = []

        for claim in claims:
            evidence: list[Source] = await self.retrieve_evidence(claim)
            best_evidence, confidence = self.verify_claim(claim, evidence)
            results.append(
                Result(claim=claim, evidence=best_evidence, confidence=confidence),
            )

        return results


async def main() -> None:
    rich.print("Beginning analysis...")
    fact_checker = AIFactChecker()
    website_url = "https://www.cnn.com/2025/02/09/business/trump-tariffs-steel-aluminum/index.html"
    rich.print(website_url)
    results = await fact_checker.analyze(website_url)

    for result in results:
        rich.print(f"Claim: {result.claim}")
        rich.print(f"Evidence: {result.evidence}")
        rich.print(f"Confidence: {result.confidence:.2f}")
        rich.print("-" * 50)


if __name__ == "__main__":
    import trio

    trio.run(main)
