"""
Job matching & ranking engine.

Matching strategy (best available wins):
  1. OpenAI text-embedding-3-small  — highest quality, requires API key
  2. sentence-transformers (all-MiniLM-L6-v2)  — good quality, local, ~80 MB model
  3. TF-IDF cosine similarity  — always available, zero dependencies beyond numpy

Final score blends semantic similarity + skill-overlap bonus + preference bonuses.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from .models import Job, RankedJob, UserProfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = _HTML_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


_STOP = frozenset(
    "a an the and or but in on at to for of with is are was were be been "
    "have has had do does did will would could should may might shall can "
    "need from by as this that these those it its i me my we our you your "
    "he she they them their who which what when where how if not".split()
)

def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOP and len(w) > 2}


# ─────────────────────────────────────────────────────────────────
# Skill matching
# ─────────────────────────────────────────────────────────────────

def skill_overlap(profile_skills: List[str], job_text: str) -> Tuple[List[str], List[str]]:
    """Return (matched_skills, missing_skills) lists."""
    job_lower = job_text.lower()
    matched = [s for s in profile_skills if s.lower() in job_lower]
    missing = [s for s in profile_skills if s.lower() not in job_lower]
    return matched, missing


# ─────────────────────────────────────────────────────────────────
# Embedding back-ends
# ─────────────────────────────────────────────────────────────────

class _OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # lazy import
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            truncated = [t[:8000] for t in texts]
            out: List[List[float]] = []
            batch_size = 100
            for i in range(0, len(truncated), batch_size):
                resp = self._client.embeddings.create(
                    model=self._model, input=truncated[i : i + batch_size]
                )
                out.extend(item.embedding for item in resp.data)
            return np.array(out, dtype=np.float32)
        except Exception as exc:
            logger.error("OpenAI embeddings failed: %s", exc)
            return None


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
            self._model = SentenceTransformer(model_name)
            logger.info("Loaded sentence-transformer: %s", model_name)
        except Exception as exc:
            logger.warning("sentence-transformers unavailable: %s", exc)
            self._model = None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            logger.error("Sentence-transformer encode failed: %s", exc)
            return None


class _TFIDFEmbedder:
    """Pure-numpy TF-IDF embedder – zero extra dependencies."""

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        tokenised = [_tokens(t) for t in corpus]
        vocab = {w for tokens in tokenised for w in tokens}
        vocab_list = sorted(vocab)
        idx = {w: i for i, w in enumerate(vocab_list)}
        n = len(corpus)
        v = len(vocab_list)

        tf = np.zeros((n, v), dtype=np.float32)
        for row, tokens in enumerate(tokenised):
            for w in tokens:
                tf[row, idx[w]] += 1
            if tf[row].sum() > 0:
                tf[row] /= tf[row].sum()

        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((n + 1) / (df + 1)) + 1.0
        return tf * idf

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        return self.fit_transform(texts)


# ─────────────────────────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────
# Public matcher
# ─────────────────────────────────────────────────────────────────

class JobMatcher:
    """
    Rank a list of :class:`Job` objects against a :class:`UserProfile`.

    Args:
        openai_api_key: If set, use OpenAI embeddings (highest accuracy).
        openai_embedding_model: OpenAI model for embeddings.
        use_sentence_transformers: If True (default), try sentence-transformers
            as the second-best option before falling back to TF-IDF.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_embedding_model: str = "text-embedding-3-small",
        use_sentence_transformers: bool = True,
    ) -> None:
        self._embedder = None

        if openai_api_key:
            try:
                self._embedder = _OpenAIEmbedder(openai_api_key, openai_embedding_model)
                logger.info("Using OpenAI embeddings for matching.")
            except Exception as exc:
                logger.warning("OpenAI embedder init failed: %s", exc)

        if self._embedder is None and use_sentence_transformers:
            st_emb = _SentenceTransformerEmbedder()
            if st_emb._model is not None:
                self._embedder = st_emb
                logger.info("Using sentence-transformers for matching.")

        if self._embedder is None:
            logger.info("Using TF-IDF matching (fallback).")
            # TF-IDF is initialised per-call because it needs the full corpus

    # ── Core ranking ─────────────────────────────────────────────

    def rank_jobs(
        self,
        profile: UserProfile,
        jobs: List[Job],
        top_k: int = 10,
    ) -> List[RankedJob]:
        """
        Score and rank ``jobs`` against ``profile``.

        Returns the top ``top_k`` :class:`RankedJob` instances sorted by
        descending match score.
        """
        if not jobs:
            return []

        profile_text = clean_text(profile.to_search_text())
        job_texts = [clean_text(job.to_match_text()) for job in jobs]

        # ── Semantic similarity ──────────────────────────────────
        sem_scores: List[float] = self._compute_semantic_scores(
            profile_text, job_texts
        )

        # ── Keyword overlap ──────────────────────────────────────
        profile_tokens = _tokens(profile_text)
        kw_scores: List[float] = []
        for jt in job_texts:
            job_tokens = _tokens(jt)
            if not profile_tokens or not job_tokens:
                kw_scores.append(0.0)
                continue
            inter = len(profile_tokens & job_tokens)
            union = len(profile_tokens | job_tokens)
            coverage = inter / len(profile_tokens)
            jaccard = inter / union if union else 0
            kw_scores.append((0.5 * jaccard + 0.5 * coverage) * 100)

        # ── Combine & apply bonuses ──────────────────────────────
        ranked: List[RankedJob] = []
        for job, sem, kw, job_text in zip(jobs, sem_scores, kw_scores, job_texts):
            base = sem * 0.65 + kw * 0.35

            matched_skills, missing_skills = skill_overlap(profile.skills, job_text)
            reasons: List[str] = []

            # Skill bonus
            if profile.skills:
                ratio = len(matched_skills) / len(profile.skills)
                base += ratio * 12
                if matched_skills:
                    reasons.append(
                        f"Matches {len(matched_skills)}/{len(profile.skills)} of your skills"
                    )

            # Desired-title bonus
            for title in profile.desired_titles:
                if title.lower() in job.title.lower():
                    base += 6
                    reasons.append(f'Matches desired role "{title}"')
                    break

            # Remote bonus
            if profile.desired_remote and job.remote:
                base += 3
                reasons.append("Remote position (as requested)")

            # Preferred location bonus
            if profile.desired_location and job.location:
                if profile.desired_location.lower() in job.location.lower():
                    base += 3
                    reasons.append(f"Located in {profile.desired_location}")

            ranked.append(
                RankedJob(
                    job=job,
                    match_score=round(min(100.0, max(0.0, base)), 1),
                    match_reasons=reasons,
                    matched_skills=matched_skills[:12],
                    missing_skills=missing_skills[:8],
                )
            )

        ranked.sort(key=lambda r: r.match_score, reverse=True)
        return ranked[:top_k]

    # ── Semantic score computation ───────────────────────────────

    def _compute_semantic_scores(
        self, profile_text: str, job_texts: List[str]
    ) -> List[float]:
        """
        Return a list of semantic similarity scores (0–100) for each job text.
        Falls back to TF-IDF if no neural embedder is available.
        """
        all_texts = [profile_text] + job_texts

        if self._embedder is not None:
            embeddings = self._embedder.encode(all_texts)
        else:
            # TF-IDF fallback
            embeddings = _TFIDFEmbedder().encode(all_texts)

        if embeddings is None or len(embeddings) == 0:
            return [0.0] * len(job_texts)

        profile_emb = embeddings[0]
        scores: List[float] = []
        for i in range(1, len(embeddings)):
            cos = _cosine(profile_emb, embeddings[i])
            # Cosine is in [-1, 1]; map to [0, 100]
            scores.append((cos + 1) / 2 * 100)
        return scores
