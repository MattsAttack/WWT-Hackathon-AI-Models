import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


class AIFactChecker:
    def __init__(self):
        # Load models
        self.claim_detection_model = pipeline("ner", model="dslim/bert-base-NER")
        self.evidence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.verification_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/t5-small-ssm-nq"
        )
        self.verification_tokenizer = AutoTokenizer.from_pretrained(
            "google/t5-small-ssm-nq"
        )

    def extract_content(self, url):
        """Scrape and clean website content."""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract core content (e.g., article text)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content

    def detect_claims(self, content):
        """Identify claims within the content."""
        sentences = content.split(".")
        claims = [
            sentence.strip() for sentence in sentences if len(sentence.strip()) > 20
        ]  # Basic length filter
        return claims

    def retrieve_evidence(self, claim):
        """Search for evidence related to a claim."""
        query_url = (
            f"https://www.googleapis.com/customsearch/v1?q={claim}&key=API_KEY&cx=CX_ID"
        )
        response = requests.get(query_url).json()
        evidence_sources = []

        for item in response.get("items", []):
            evidence_sources.append(
                {
                    "title": item["title"],
                    "snippet": item["snippet"],
                    "link": item["link"],
                }
            )
        return evidence_sources

    def verify_claim(self, claim, evidence):
        """Compare claim with evidence using semantic similarity."""
        evidence_texts = [e["snippet"] for e in evidence]
        similarities = []

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

    def analyze(self, url):
        """End-to-end analysis: Extract, detect claims, and fact-check."""
        content = self.extract_content(url)
        claims = self.detect_claims(content)
        results = []

        for claim in claims:
            evidence = self.retrieve_evidence(claim)
            best_evidence, confidence = self.verify_claim(claim, evidence)
            results.append(
                {"claim": claim, "evidence": best_evidence, "confidence": confidence}
            )

        return results


if __name__ == "__main__":
    print("this part is running")
    fact_checker = AIFactChecker()
    website_url = "https://www.cnn.com/2025/02/09/business/trump-tariffs-steel-aluminum/index.html"
    print(website_url)
    results = fact_checker.analyze(website_url)

    for result in results:
        print(f"Claim: {result['claim']}")
        print(f"Evidence: {result['evidence']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 50)
