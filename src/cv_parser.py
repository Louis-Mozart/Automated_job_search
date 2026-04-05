"""
CV / Resume parser.

Supports:
  • PDF (via pdfplumber)
  • DOCX / DOC (via python-docx)
  • Plain text / Markdown

Profile extraction:
  • OpenAI LLM (if API key supplied)  ← best quality
  • Regex + keyword heuristics        ← always-available fallback
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from .models import UserProfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Text extraction helpers
# ─────────────────────────────────────────────────────────────────

def _extract_pdf(content: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except ImportError:
        logger.error("pdfplumber is not installed.")
        return ""
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        return ""


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.error("python-docx is not installed.")
        return ""
    except Exception as exc:
        logger.error("DOCX extraction failed: %s", exc)
        return ""


def extract_text(content: bytes, filename: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(content)
    if ext in {".docx", ".doc"}:
        return _extract_docx(content)
    # Plain text / markdown / unknown
    return content.decode("utf-8", errors="ignore")


# ─────────────────────────────────────────────────────────────────
# Regex / heuristic extraction (no LLM required)
# ─────────────────────────────────────────────────────────────────

# fmt: off
_TECH_SKILLS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "Bash", "Shell",
    # Web / frameworks
    "React", "Vue", "Angular", "Next.js", "Svelte", "Node.js",
    "Django", "FastAPI", "Flask", "Spring", "Express", "Rails",
    # Data / ML / AI
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Artificial Intelligence", "Reinforcement Learning", "Generative AI",
    "Large Language Models", "LLM", "Transformer", "BERT", "GPT",
    "Statistical Modeling", "Data Analysis", "Research", "Scientific Computing",
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Hugging Face",
    "Pandas", "NumPy", "SciPy", "Spark", "Kafka", "Airflow", "dbt",
    "Jupyter", "Matplotlib", "Seaborn", "OpenCV",
    # Cloud / DevOps
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "CI/CD", "Jenkins", "GitHub Actions", "Ansible", "Helm",
    # Databases
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Cassandra", "DynamoDB", "BigQuery", "Snowflake",
    # General
    "REST API", "GraphQL", "Microservices", "Linux", "Git",
    "Agile", "Scrum", "Jira", "DevOps", "MLOps",
]
# fmt: on

# Patterns that suggest a research / scientist / analyst job seeker
_RESEARCH_TITLE_RE = re.compile(
    r"\b(research(?:er)?|scientist|r\s*&\s*d|r\s+and\s+d|data\s+scientist"
    r"|machine\s+learning\s+engineer|ai\s+engineer|ml\s+engineer"
    r"|research\s+engineer|research\s+scientist|phd|professor"
    r"|computational\s+biologist|bioinformatics|quantitative\s+researcher)\b",
    re.IGNORECASE,
)

# Map detected skills / title keywords → better search queries
_QUERY_TEMPLATES = [
    # (trigger keywords in description, search query to generate)
    (["artificial intelligence", "machine learning", "research"],
     "AI machine learning researcher"),
    (["deep learning", "neural", "research"],
     "deep learning research scientist"),
    (["nlp", "natural language"],
     "NLP researcher machine learning"),
    (["computer vision", "image recognition"],
     "computer vision researcher"),
    (["data science", "data scientist"],
     "data scientist machine learning"),
    (["reinforcement learning"],
     "reinforcement learning researcher"),
    (["bioinformatics", "computational biology"],
     "computational biology researcher"),
    (["quantitative", "statistics", "econometrics"],
     "quantitative researcher data scientist"),
]

_EXP_RE = [
    r"(\d+)\+?\s+years?\s+of\s+(?:professional\s+)?experience",
    r"(\d+)\+?\s+years?\s+experience",
    r"experience\s+of\s+(\d+)\+?\s+years?",
    r"(\d+)\+?\s+yrs?\s+(?:of\s+)?experience",
]


def _parse_with_regex(text: str) -> UserProfile:
    """Best-effort profile extraction without an LLM."""
    text_lower = text.lower()

    # Skills
    skills = [s for s in _TECH_SKILLS if s.lower() in text_lower]

    # Experience years
    exp_years: Optional[int] = None
    for pattern in _EXP_RE:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            exp_years = int(m.group(1))
            break

    # Email / phone
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    phones = re.findall(r"\+?[\d][\d\s\-().]{7,14}[\d]", text)

    # Desired titles via research pattern
    desired_titles: List[str] = []
    title_match = _RESEARCH_TITLE_RE.search(text)
    if title_match:
        t = title_match.group(0).lower().strip()
        desired_titles = [t.title()]

    # Build search queries — try to match a template first
    queries: List[str] = []
    for triggers, query_str in _QUERY_TEMPLATES:
        if sum(1 for kw in triggers if kw in text_lower) >= 2:
            queries.append(query_str)

    # Fallback: use top detected skills
    if not queries and skills:
        # Exclude single-letter skills (like "R") when building a query
        meaningful = [s for s in skills if len(s) > 1 and s not in ("R", "Go")]
        if meaningful:
            queries.append(" ".join(meaningful[:5]))

    # Last resort: first meaningful words of the text
    if not queries:
        words = re.findall(r"[a-zA-Z]{4,}", text)[:10]
        queries = [" ".join(words)]

    # Keywords for matching = skills + query words
    keywords = list(dict.fromkeys(skills + re.findall(r"[a-zA-Z]{4,}", text)[:30]))

    return UserProfile(
        raw_text=text,
        email=emails[0] if emails else None,
        phone=phones[0].strip() if phones else None,
        skills=skills,
        experience_years=exp_years,
        desired_titles=desired_titles,
        keywords=keywords[:25],
        search_queries=queries[:5],
    )


# ─────────────────────────────────────────────────────────────────
# LLM-based extraction (OpenAI)
# ─────────────────────────────────────────────────────────────────

_LLM_PROMPT = """\
You are an expert HR assistant. Extract structured information from the text below.
Return ONLY a valid JSON object (no markdown fences) with these fields:

{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "skills": [list of technical and soft skills],
  "experience_years": integer or null,
  "current_title": string or null,
  "desired_titles": [job titles the candidate is looking for],
  "desired_location": string or null,
  "desired_remote": boolean,
  "education": [list of degree/school strings],
  "certifications": [list],
  "languages": [spoken or programming languages],
  "summary": "2-3 sentence professional summary",
  "keywords": [top 20 keywords for job search],
  "search_queries": [3-5 ready-to-use job-board search strings]
}

TEXT:
"""


def _parse_with_llm(text: str, api_key: str, model: str) -> UserProfile:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _LLM_PROMPT + text[:4500]}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data: dict = json.loads(response.choices[0].message.content)

        return UserProfile(
            raw_text=text,
            name=data.get("name"),
            email=data.get("email"),
            phone=data.get("phone"),
            skills=data.get("skills") or [],
            experience_years=data.get("experience_years"),
            current_title=data.get("current_title"),
            desired_titles=data.get("desired_titles") or [],
            desired_location=data.get("desired_location"),
            desired_remote=bool(data.get("desired_remote", False)),
            education=data.get("education") or [],
            certifications=data.get("certifications") or [],
            languages=data.get("languages") or [],
            summary=data.get("summary") or "",
            keywords=data.get("keywords") or [],
            search_queries=data.get("search_queries") or [],
        )
    except Exception as exc:
        logger.warning("LLM parsing failed (%s). Falling back to regex.", exc)
        return _parse_with_regex(text)


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

class CVParser:
    """
    Parse a CV file or freeform text into a :class:`UserProfile`.

    Args:
        openai_api_key: If provided, uses GPT for high-quality extraction.
        model: OpenAI chat model to use.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.openai_api_key = openai_api_key
        self.model = model

    # ── File upload ──────────────────────────────────────────────

    def parse_file(self, content: bytes, filename: str) -> UserProfile:
        """Parse an uploaded file (PDF / DOCX / TXT)."""
        text = extract_text(content, filename)
        if not text.strip():
            raise ValueError(
                f"Could not extract any text from '{filename}'. "
                "Please ensure the file is not empty or password-protected."
            )
        return self.parse_text(text)

    # ── Freeform text ────────────────────────────────────────────

    def parse_text(self, text: str) -> UserProfile:
        """Parse freeform text (job description, pasted CV, etc.)."""
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")

        if self.openai_api_key:
            return _parse_with_llm(text, self.openai_api_key, self.model)
        return _parse_with_regex(text)
