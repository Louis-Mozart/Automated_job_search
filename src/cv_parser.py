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

# General (non-tech) skills – hospitality, education, trades, etc.
_GENERAL_SKILLS = [
    # Hospitality / Food & Beverage
    "Waiter", "Waitress", "Server", "Barista", "Bartender", "Chef", "Cook",
    "Kitchen Staff", "Dishwasher", "Catering", "Hospitality", "Food Service",
    "Food Preparation", "Restaurant", "Café",
    # Cleaning / Facility
    "Cleaning", "Housekeeping", "Janitor", "Maintenance", "Sanitation",
    "Maid", "Car Wash", "Auto Detailing", "Facility Management",
    # Education / Teaching
    "Teaching", "Teacher", "Tutor", "Tutoring", "Instructor", "Educator",
    "Childcare", "Kindergarten", "Nursery", "Babysitting",
    "Special Education", "Primary Education", "Secondary Education",
    # Driving / Logistics
    "Driving", "Driver", "Delivery", "Courier", "Logistics", "Transport",
    "Forklift", "Warehouse", "Picking", "Packing", "Dispatching",
    # Retail / Sales
    "Retail", "Cashier", "Sales Assistant", "Shop Assistant", "Store Clerk",
    "Customer Service", "Customer Support", "Sales", "Merchandising",
    # Healthcare / Care
    "Caregiver", "Care Worker", "Nursing Assistant", "Home Help",
    "Support Worker", "Social Work", "Elderly Care",
    # Security
    "Security Guard", "Security Officer", "Doorman", "Bouncer",
    # Trades
    "Electrician", "Plumber", "Carpenter", "Painter", "Decorator",
    "Handyman", "Mechanic", "Auto Technician", "Welder", "Construction",
    # Gardening / Outdoor
    "Gardening", "Gardener", "Landscaping", "Groundskeeper",
    # Admin / Office (non-IT)
    "Receptionist", "Administrative Assistant", "Secretary", "Data Entry",
    "Office Assistant",
]

# Combined list used for skill extraction
_ALL_SKILLS = _TECH_SKILLS + _GENERAL_SKILLS
# fmt: on

# Patterns that suggest a research / scientist / analyst job seeker
_RESEARCH_TITLE_RE = re.compile(
    r"\b(research(?:er)?|scientist|r\s*&\s*d|r\s+and\s+d|data\s+scientist"
    r"|machine\s+learning\s+engineer|ai\s+engineer|ml\s+engineer"
    r"|research\s+engineer|research\s+scientist|phd|professor"
    r"|computational\s+biologist|bioinformatics|quantitative\s+researcher)\b",
    re.IGNORECASE,
)

# Detect non-tech / service / trade job titles
_GENERAL_TITLE_RE = re.compile(
    r"\b(waiter|waitress|server|barista|bartender|chef|sous\s+chef|cook"
    r"|kitchen\s+(?:staff|porter|hand|worker)|dishwasher"
    r"|cleaner|cleaning\s+(?:staff|lady|person)|housekeeper|housekeeping"
    r"|janitor|car\s+(?:cleaner|wash)\s*(?:attendant)?|auto\s+detailer"
    r"|teacher|tutor|instructor|educator|lecturer"
    r"|childcare\s*(?:worker)?|kindergarten\s*(?:teacher)?|nursery\s*(?:nurse)?"
    r"|driver|delivery\s*(?:driver)?|courier|dispatcher"
    r"|warehouse\s*(?:worker|operative|associate)?|picker|packer"
    r"|forklift\s*(?:operator|driver)?|logistics\s*(?:coordinator|worker)?"
    r"|cashier|sales\s*(?:assistant|associate|representative)?"
    r"|shop\s*(?:assistant|keeper)?|store\s*(?:clerk|associate|assistant)?"
    r"|retail\s*(?:worker|associate)?"
    r"|customer\s*(?:service|support)\s*(?:representative|agent)?"
    r"|receptionist|administrative\s*(?:assistant)?|secretary|office\s*(?:assistant)?"
    r"|caregiver|care\s*(?:worker|giver|assistant)|nursing\s*assistant"
    r"|home\s*(?:help|carer)|support\s*worker|social\s*(?:worker)?"
    r"|security\s*(?:guard|officer)|doorman|bouncer"
    r"|electrician|plumber|carpenter|painter|decorator|handyman|mechanic|welder"
    r"|gardener|landscaper|groundskeeper)\b",
    re.IGNORECASE,
)

# Pattern: "want to work as a X", "looking for a X", etc.
_WORK_AS_RE = re.compile(
    r"\b(?:work(?:ing)?\s+as\s+(?:a\s+|an\s+)?"
    r"|looking\s+for\s+(?:a\s+|an\s+)?"
    r"|want\s+to\s+(?:work\s+as\s+(?:a\s+|an\s+)?|be\s+(?:a\s+|an\s+)?)?"
    r"|seeking\s+(?:a\s+|an\s+)?"
    r"|job\s+as\s+(?:a\s+|an\s+)?"
    r"|position\s+(?:as\s+)?(?:a\s+|an\s+)?)"
    r"([a-z][a-z\-\s]{2,30}?)(?=\s+in\s|\s+at\s|\s+for\s|\s+position|\s+job|\s+role|[,.\n]|$)",
    re.IGNORECASE,
)

# Map detected skills / title keywords → better search queries
# Each entry: (trigger_keywords, search_query, min_matches_required)
_QUERY_TEMPLATES = [
    # ── Tech / Research ────────────────────────────────────────
    (["artificial intelligence", "machine learning", "research"],
     "AI machine learning researcher", 2),
    (["deep learning", "neural", "research"],
     "deep learning research scientist", 2),
    (["nlp", "natural language"],
     "NLP researcher machine learning", 1),
    (["computer vision", "image recognition"],
     "computer vision researcher", 1),
    (["data science", "data scientist"],
     "data scientist machine learning", 1),
    (["reinforcement learning"],
     "reinforcement learning researcher", 1),
    (["bioinformatics", "computational biology"],
     "computational biology researcher", 1),
    (["quantitative", "statistics", "econometrics"],
     "quantitative researcher data scientist", 2),
    # ── Hospitality / Food & Beverage ─────────────────────────
    (["waiter", "waitress", "restaurant", "food service",
      "hospitality", "barista", "server", "café", "cafe"],
     "waiter server restaurant hospitality", 1),
    (["chef", "cook", "kitchen", "culinary", "sous chef", "cooking"],
     "chef cook kitchen food preparation", 1),
    (["bartender", "cocktail", "mixing drinks"],
     "bartender bar staff mixologist", 1),
    # ── Cleaning / Facility ────────────────────────────────────
    (["cleaning", "cleaner", "housekeeping", "janitor",
      "car wash", "car cleaner", "auto detail", "maid"],
     "cleaner housekeeping facility maintenance", 1),
    # ── Education / Teaching ───────────────────────────────────
    (["teacher", "teaching", "education", "school",
      "tutor", "tutoring", "instructor", "educator"],
     "teacher educator school tutor", 1),
    (["childcare", "kindergarten", "nursery", "babysitting", "nanny"],
     "childcare worker kindergarten nursery teacher", 1),
    # ── Driving / Logistics ────────────────────────────────────
    (["driver", "delivery driver", "courier", "transport",
      "logistics", "forklift", "warehouse", "dispatch"],
     "driver delivery logistics warehouse", 1),
    # ── Retail / Customer service ──────────────────────────────
    (["retail", "cashier", "sales assistant", "shop assistant", "store clerk"],
     "retail assistant cashier customer service", 1),
    (["customer service", "customer support", "call center", "helpdesk"],
     "customer service representative support agent", 1),
    # ── Security ──────────────────────────────────────────────
    (["security guard", "security officer", "doorman", "bouncer"],
     "security guard officer", 1),
    # ── Healthcare / Care ─────────────────────────────────────
    (["caregiver", "care worker", "nursing assistant", "home help", "elderly care"],
     "caregiver care worker nursing assistant", 1),
    # ── Trades ────────────────────────────────────────────────
    (["electrician", "electrical wiring", "electrical installation"],
     "electrician electrical technician", 1),
    (["plumber", "plumbing", "pipe installation"],
     "plumber plumbing technician", 1),
    (["mechanic", "automotive repair", "vehicle repair"],
     "mechanic auto technician", 1),
    (["carpenter", "woodwork", "joinery"],
     "carpenter woodworker construction", 1),
    (["painter", "decorator", "painting", "decorating"],
     "painter decorator", 1),
    # ── Administrative ────────────────────────────────────────
    (["receptionist", "administrative assistant", "office assistant", "secretary"],
     "receptionist administrative assistant office", 1),
    (["gardener", "landscaping", "groundskeeper", "gardening"],
     "gardener landscaper grounds maintenance", 1),
]

_EXP_RE = [
    r"(\d+)\+?\s+years?\s+of\s+(?:professional\s+)?experience",
    r"(\d+)\+?\s+years?\s+experience",
    r"experience\s+of\s+(\d+)\+?\s+years?",
    r"(\d+)\+?\s+yrs?\s+(?:of\s+)?experience",
]

# Pre-compile word-boundary patterns for short skills (≤ 3 chars)
# to avoid false positives like "R" matching inside "waiter".
_SHORT_SKILL_RE: dict[str, re.Pattern] = {
    s: re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)
    for s in _ALL_SKILLS if len(s) <= 3
}


def _skill_in_text(skill: str, text_lower: str, text_orig: str) -> bool:
    """Return True if *skill* appears in the text as a proper word/phrase."""
    if skill in _SHORT_SKILL_RE:
        return bool(_SHORT_SKILL_RE[skill].search(text_orig))
    return skill.lower() in text_lower


def _parse_with_regex(text: str) -> UserProfile:
    """Best-effort profile extraction without an LLM."""
    text_lower = text.lower()

    # Skills — check both tech and general lists (word-boundary safe)
    skills = [s for s in _ALL_SKILLS if _skill_in_text(s, text_lower, text)]

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

    # ── Desired titles ───────────────────────────────────────────
    desired_titles: List[str] = []

    # 1. Explicit "work as X" / "looking for X" pattern
    work_as_match = _WORK_AS_RE.search(text)
    if work_as_match:
        candidate = work_as_match.group(1).strip().lower()
        # Sanity: at least 3 chars and not a filler word
        if len(candidate) >= 3 and candidate not in {"job", "work", "role", "position"}:
            desired_titles.append(candidate.title())

    # 2. Research / scientist / analyst title pattern
    if not desired_titles:
        title_match = _RESEARCH_TITLE_RE.search(text)
        if title_match:
            desired_titles.append(title_match.group(0).strip().title())

    # 3. General service / trade title pattern
    if not desired_titles:
        title_match = _GENERAL_TITLE_RE.search(text)
        if title_match:
            desired_titles.append(title_match.group(0).strip().title())

    # ── Search queries via templates ─────────────────────────────
    queries: List[str] = []
    for triggers, query_str, min_matches in _QUERY_TEMPLATES:
        if sum(1 for kw in triggers if kw in text_lower) >= min_matches:
            queries.append(query_str)
            if len(queries) >= 3:  # avoid over-generating
                break

    # Fallback 1: use desired title if we have one
    if not queries and desired_titles:
        queries.append(desired_titles[0])

    # Fallback 2: top detected skills (skip trivial ones)
    if not queries and skills:
        meaningful = [s for s in skills if len(s) > 1 and s not in ("R", "Go")]
        if meaningful:
            queries.append(" ".join(meaningful[:5]))

    # Last resort: extract most meaningful words from the text
    if not queries:
        # Prefer capitalised words (likely nouns / proper names / job titles)
        cap_words = re.findall(r"\b[A-ZÜÖÄ][a-zA-ZüöäÜÖÄß]{3,}\b", text)
        words = cap_words[:6] if cap_words else re.findall(r"[a-zA-Z]{4,}", text)[:8]
        queries = [" ".join(words)] if words else ["job opening"]

    # Keywords for matching = skills + query words
    keywords = list(dict.fromkeys(
        skills + re.findall(r"[a-zA-Z]{4,}", text)[:30]
    ))

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
