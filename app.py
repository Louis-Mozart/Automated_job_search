"""
AutoJob Finder – Streamlit application entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()   # picks up .env file if present

from config import settings
from src.cv_parser import CVParser
from src.job_matcher import JobMatcher
from src.job_searcher import JobSearcher
from src.location_filter import score_location_match
from src.models import RankedJob, UserProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoJob Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Header gradient banner ── */
.banner {
    text-align: center;
    padding: 2.2rem 1rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}
.banner h1 { margin: 0; font-size: 2.4rem; }
.banner p  { margin: .4rem 0 0; opacity: .85; font-size: 1.1rem; }

/* ── Score bubble ── */
.score-bubble {
    text-align: center;
    padding: 12px 8px;
    border-radius: 12px;
    border: 2.5px solid;
}
.score-bubble .num  { font-size: 2rem; font-weight: 800; }
.score-bubble .lbl  { font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }

/* ── Pill tags ── */
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: .78rem;
    margin: 2px;
    font-weight: 500;
}
.tag-skill   { background:#dbeafe; color:#1d4ed8; }
.tag-missing { background:#fee2e2; color:#b91c1c; }
.tag-remote  { background:#d1fae5; color:#065f46; }
.tag-type    { background:#f3f4f6; color:#374151; }
.tag-source  { background:#fef3c7; color:#92400e; }

/* ── Section divider ── */
hr.thin { margin: .6rem 0; border: none; border-top: 1px solid #e5e7eb; }

/* ── Make primary button full-width in column ── */
div[data-testid="stButton"] > button { border-radius: 8px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────

def _ss(key: str, default=None):
    """Get a value from Streamlit session state with a default."""
    return st.session_state.get(key, default)


def _reset_results() -> None:
    st.session_state["results"] = None
    st.session_state["profile"] = None
    st.session_state["user_query"] = None
    # Also clear any in-progress refinement
    st.session_state.pop("pending_search_text", None)
    st.session_state.pop("draft_query", None)


def rephrase_query(raw_text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """
    Use an LLM to rewrite a freeform user description into a clean,
    focused job-search query (2-4 sentences, no filler).
    """
    from openai import OpenAI

    prompt = (
        "You are a job search assistant. The user has written a freeform description "
        "of the job they are looking for. Rewrite it as a clean, focused 2-4 sentence "
        "job search description that will work well for matching against job listings.\n\n"
        "Rules:\n"
        "- Keep: desired job title(s), key skills, experience level, location, requirements.\n"
        "- Remove: filler words, personal rambling, irrelevant details.\n"
        "- Do NOT invent or assume details not mentioned by the user.\n"
        "- Write in a neutral, professional tone.\n\n"
        f"User's text:\n{raw_text[:3000]}\n\n"
        "Rephrased description:"
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────
# Service factories
# ─────────────────────────────────────────────────────────────────

def _build_services():
    """Instantiate CVParser, JobSearcher, and JobMatcher from current session keys."""
    openai_key   = _ss("openai_key")   or settings.openai_api_key
    rapid_key    = _ss("rapidapi_key") or settings.rapidapi_key
    adzuna_id    = _ss("adzuna_id")    or settings.adzuna_app_id
    adzuna_key   = _ss("adzuna_key")   or settings.adzuna_app_key
    country      = _ss("adzuna_country", settings.adzuna_country)
    jooble_key   = _ss("jooble_key")   or settings.jooble_api_key

    parser   = CVParser(openai_api_key=openai_key, model=settings.openai_model)
    searcher = JobSearcher(
        rapidapi_key=rapid_key,
        adzuna_app_id=adzuna_id,
        adzuna_app_key=adzuna_key,
        adzuna_country=country,
        jooble_api_key=jooble_key,
    )
    matcher  = JobMatcher(
        openai_api_key=openai_key,
        openai_embedding_model=settings.openai_embedding_model,
    )
    return parser, searcher, matcher


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[Optional[str], bool, int]:
    """Render sidebar controls; return (location, remote_only, k)."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # ── API Keys ──────────────────────────────────────────────
        with st.expander("🔑 API Keys", expanded=False):
            st.info(
                "**No keys needed to start!** The app always uses three free sources: "
                "**WeWorkRemotely** (RSS), **HN Hiring** (monthly thread, great for AI/ML), "
                "and **Remotive**. Add keys below for LinkedIn/Indeed/Glassdoor coverage."
            )
            st.session_state["openai_key"] = st.text_input(
                "OpenAI API Key",
                value=_ss("openai_key", settings.openai_api_key or ""),
                type="password",
                help="Smart CV parsing + semantic matching. Get one at platform.openai.com",
            )
            st.session_state["rapidapi_key"] = st.text_input(
                "RapidAPI Key  (JSearch)",
                value=_ss("rapidapi_key", settings.rapidapi_key or ""),
                type="password",
                help="Aggregates LinkedIn, Indeed, Glassdoor & more. rapidapi.com → JSearch",
            )
            st.markdown("**Adzuna** – free tier at developer.adzuna.com")
            st.session_state["adzuna_id"] = st.text_input(
                "Adzuna App ID",
                value=_ss("adzuna_id", settings.adzuna_app_id or ""),
                type="password",
            )
            st.session_state["adzuna_key"] = st.text_input(
                "Adzuna App Key",
                value=_ss("adzuna_key", settings.adzuna_app_key or ""),
                type="password",
            )
            st.session_state["adzuna_country"] = st.selectbox(
                "Adzuna country",
                options=["us", "gb", "de", "fr", "au", "ca", "nl", "sg"],
                index=["us", "gb", "de", "fr", "au", "ca", "nl", "sg"].index(
                    _ss("adzuna_country", settings.adzuna_country)
                ),
            )

        st.markdown("---")
        st.markdown("## 🔍 Search Preferences")

        location = st.text_input(
            "Preferred Location",
            placeholder="e.g. New York, London, Berlin …",
            key="pref_location",
        ) or None

        remote = st.checkbox("Remote positions only", key="pref_remote")

        k = st.slider(
            "Number of top jobs to return (k)",
            min_value=1,
            max_value=100,
            value=settings.default_k,
            key="pref_k",
        )

        st.markdown("---")
        st.caption("Made with ❤️ by KTLM using Streamlit · AutoJob Finder")

    return location, remote, k


# ─────────────────────────────────────────────────────────────────
# Job card renderer
# ─────────────────────────────────────────────────────────────────

def render_job_card(rj: RankedJob, rank: int) -> None:
    """Render a single job result card."""
    job   = rj.job
    score = rj.match_score
    color = rj.score_color

    with st.container():
        col_info, col_score = st.columns([0.82, 0.18])

        # ── Left: job info ──────────────────────────────────────
        with col_info:
            st.markdown(f"### {rank}. {job.title}")

            meta_parts = [f"🏢 **{job.company}**"]
            if job.location:
                meta_parts.append(f"📍 {job.location}")
            st.markdown("&nbsp;&nbsp;".join(meta_parts))

            # Salary
            sal = job.get_salary_display()
            if sal != "Not disclosed":
                st.markdown(f"💰 {sal}")

            # Tag pills
            tags_html = ""
            if job.remote:
                tags_html += '<span class="tag tag-remote">🌍 Remote</span>'
            if job.job_type:
                label = job.job_type.replace("_", " ").replace("-", " ").title()
                tags_html += f'<span class="tag tag-type">{label}</span>'
            tags_html += f'<span class="tag tag-source">via {job.source}</span>'
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

        # ── Right: score bubble ─────────────────────────────────
        with col_score:
            st.markdown(
                f'<div class="score-bubble" style="border-color:{color}; '
                f'background:{color}18;">'
                f'<div class="num" style="color:{color};">{score:.0f}%</div>'
                f'<div class="lbl" style="color:{color};">{rj.score_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Skills ──────────────────────────────────────────────
        if rj.matched_skills or rj.missing_skills:
            c1, c2 = st.columns(2)
            with c1:
                if rj.matched_skills:
                    pills = "".join(
                        f'<span class="tag tag-skill">✅ {s}</span>'
                        for s in rj.matched_skills[:10]
                    )
                    st.markdown(f"**Matched skills:** {pills}", unsafe_allow_html=True)
            with c2:
                if rj.missing_skills:
                    pills = "".join(
                        f'<span class="tag tag-missing">📚 {s}</span>'
                        for s in rj.missing_skills[:6]
                    )
                    st.markdown(f"**To develop:** {pills}", unsafe_allow_html=True)

        # ── Match reasons ───────────────────────────────────────
        if rj.match_reasons:
            st.markdown("**Why it matches:** " + " &bull; ".join(rj.match_reasons),
                        unsafe_allow_html=True)

        # ── Expandable description ──────────────────────────────
        if job.description:
            with st.expander("📄 Full job description"):
                desc = re.sub(r"<[^>]+>", " ", job.description)
                desc = re.sub(r"\s+", " ", desc).strip()
                st.markdown(desc[:3500] + ("…" if len(desc) > 3500 else ""))

        # ── Apply button ─────────────────────────────────────────
        if job.url:
            st.link_button("🚀 Apply Now", job.url, type="primary")
        else:
            st.caption("No direct application link available.")

        st.markdown('<hr class="thin">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Search flow
# ─────────────────────────────────────────────────────────────────

def run_search(
    parser: CVParser,
    searcher: JobSearcher,
    matcher: JobMatcher,
    uploaded_file,
    description_text: str,
    location: Optional[str],
    remote: bool,
    k: int,
) -> None:
    """
    Full pipeline: parse → search → rank.
    Writes results to session state.
    """
    progress = st.progress(0, text="Starting …")
    status   = st.empty()

    # Save raw user query so results page can display it
    st.session_state["user_query"] = (
        description_text.strip() if description_text.strip()
        else (f"📄 {uploaded_file.name}" if uploaded_file else "")
    )

    try:
        # ── 1. Parse profile ─────────────────────────────────────
        status.info("📋 Analysing your profile …")
        progress.progress(10, text="Parsing profile …")

        if uploaded_file is not None:
            profile = parser.parse_file(uploaded_file.read(), uploaded_file.name)
        else:
            profile = parser.parse_text(description_text)

        # Allow user to override location / remote
        if location and not profile.desired_location:
            profile.desired_location = location
        if remote:
            profile.desired_remote = True

        st.session_state["profile"] = profile
        progress.progress(28, text="Profile analysed.")

        # Show extracted profile
        with st.expander("🎯 Extracted Profile — click to review", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                if profile.name:
                    st.markdown(f"**Name:** {profile.name}")
                if profile.current_title:
                    st.markdown(f"**Current title:** {profile.current_title}")
                if profile.experience_years:
                    st.markdown(f"**Experience:** {profile.experience_years} yrs")
                if profile.desired_titles:
                    st.markdown(f"**Target roles:** {', '.join(profile.desired_titles[:4])}")
                if profile.desired_location:
                    st.markdown(f"**Location pref:** {profile.desired_location}")
                if profile.desired_remote:
                    st.markdown("**Wants remote:** Yes")
            with c2:
                if profile.skills:
                    pills = "".join(
                        f'<span class="tag tag-skill">{s}</span>'
                        for s in profile.skills[:20]
                    )
                    st.markdown(f"**Skills:**<br>{pills}", unsafe_allow_html=True)

        # ── 2. Search ────────────────────────────────────────────
        src_names = ", ".join(searcher.available_sources)
        is_free_only = not searcher.has_paid_sources
        status.info(f"🔍 Searching jobs on: **{src_names}** …")
        progress.progress(42, text="Fetching jobs …")

        if is_free_only:
            location_set = bool(location or profile.desired_location)
            loc_display = location or profile.desired_location or ""
            # Rough check: is the desired location inside Germany/DACH?
            _DACH = {"germany", "deutschland", "munich", "münchen", "berlin",
                     "hamburg", "frankfurt", "cologne", "köln", "stuttgart",
                     "düsseldorf", "austria", "wien", "vienna", "switzerland",
                     "zürich", "zurich", "bern", "graz", "salzburg"}
            loc_is_dach = any(w in loc_display.lower() for w in _DACH)
            st.info(
                "ℹ️ **Free-tier mode** — active sources:\n"
                "- **WeWorkRemotely, HN Hiring, Remotive** – remote tech jobs (~400)\n"
                "- **Bundesagentur** – all job types in Germany "
                "(waiter, cleaner, teacher…) – best for German cities\n\n"
                + (
                    f"📍 **{loc_display}** looks like a German city → "
                    "Bundesagentur results included above. "
                    if loc_is_dach and location_set else
                    f"📍 **Non-tech jobs outside Germany** (e.g. Paris): add a free "
                    "[Jooble key](https://jooble.org/api/index) or "
                    "an [Adzuna key](https://developer.adzuna.com) with the right country. "
                    if location_set and not loc_is_dach else ""
                )
                + "For LinkedIn/Indeed/Glassdoor add a **RapidAPI** key."
            )

        jobs = searcher.search(
            profile=profile,
            location=location or profile.desired_location,
            remote=remote or profile.desired_remote,
            max_results=max(settings.max_jobs_to_fetch, k * 8),
        )
        progress.progress(72, text=f"Fetched {len(jobs)} jobs.")

        if not jobs:
            status.empty()
            progress.empty()
            st.warning(
                "**No jobs were fetched.** This usually means a network error "
                "reaching the job sources.  \n\n"
                "**Try:**  \n"
                "- Check your internet connection  \n"
                "- Add a RapidAPI or Adzuna key in the sidebar for paid sources  \n"
                "- Wait a moment and search again (free sources may be rate-limiting)"
            )
            return

        st.info(f"Found **{len(jobs)}** jobs — ranking top **{k}** matches …")

        # ── 3. Rank ──────────────────────────────────────────────
        status.info("🧠 Ranking by match score …")
        progress.progress(88, text="Ranking …")

        ranked = matcher.rank_jobs(profile, jobs, top_k=k)
        progress.progress(100, text="Done!")
        status.success(f"✅ Done! Showing your top {len(ranked)} matches.")

        # ── Relevance sanity check ───────────────────────────────
        # The free sources only carry remote/tech jobs. If the user is looking
        # for a non-tech or in-person local role the ranked list will be filled
        # with unrelated tech jobs. Detect this and warn immediately.
        if is_free_only and ranked:
            top_score = ranked[0].match_score
            desired_str = " ".join(profile.desired_titles + (profile.search_queries or [])).lower()
            _TECH_WORDS = {
                "engineer", "developer", "software", "scientist", "analyst",
                "devops", "designer", "architect", "researcher", "consultant",
                "programmer", "data", "ml", "ai", "backend", "frontend",
                "fullstack", "cloud", "security", "product", "sre",
            }
            # Match on whole words to avoid substring hits ("ai" inside "waiter")
            desired_words = set(re.findall(r"\b\w+\b", desired_str))
            looks_non_tech = bool(desired_str) and not (desired_words & _TECH_WORDS)
            if top_score < 32 or looks_non_tech:
                titles_str = (
                    ", ".join(f'**{t}**' for t in profile.desired_titles[:2])
                    if profile.desired_titles else "this role"
                )
                loc_hint = profile.desired_location or location or "your area"
                st.warning(
                    f"⚠️ **Low relevance warning** — the free sources "
                    f"(WeWorkRemotely, HN Hiring, Remotive) **only contain "
                    f"remote tech jobs**. Results shown may not match {titles_str} "
                    f"in **{loc_hint}**.\n\n"
                    "**To find the right jobs, pick one of these options:**\n"
                    "- 🆓 **Adzuna** (250 free req/day) — "
                    "sign up at [developer.adzuna.com](https://developer.adzuna.com), "
                    "add your App ID + Key in the sidebar, and set the correct "
                    "**Adzuna country** (e.g. `de` for Germany, `fr` for France).\n"
                    "- 💳 **JSearch / RapidAPI** — aggregates Indeed, LinkedIn, "
                    "Glassdoor and ZipRecruiter worldwide.\n"
                    "- 🌐 Search directly on "
                    "[Indeed](https://indeed.com), [StepStone](https://stepstone.de), "
                    "or the job board of your country."
                )

        st.session_state["results"] = ranked
        st.rerun()

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Error: {exc}")
        logger.exception("Search pipeline error")


# ─────────────────────────────────────────────────────────────────
# Results page
# ─────────────────────────────────────────────────────────────────

def render_results(ranked_jobs: List[RankedJob]) -> None:
    profile: Optional[UserProfile] = _ss("profile")

    # ── Summary metrics ──────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    avg = sum(j.match_score for j in ranked_jobs) / len(ranked_jobs)
    remote_n = sum(1 for j in ranked_jobs if j.job.remote)
    sources   = {j.job.source for j in ranked_jobs}

    m1.metric("Jobs shown",       len(ranked_jobs))
    m2.metric("Avg match score",  f"{avg:.1f}%")
    m3.metric("Remote positions", remote_n)
    m4.metric("Job sources",      len(sources))

    st.markdown("---")

    # ── Header row ───────────────────────────────────────────────
    hc1, hc2 = st.columns([5, 1])
    with hc1:
        st.markdown(f"## Your Top {len(ranked_jobs)} Job Matches")
    with hc2:
        if st.button("🔄 New Search", type="secondary"):
            _reset_results()
            st.rerun()

    # ── Refine search query ───────────────────────────────────────
    user_query = _ss("user_query", "")
    is_file_search = user_query.startswith("\U0001f4c4 ")
    st.markdown("#### \u270f\ufe0f Refine your search")
    refine_text: str = st.text_area(
        "Refine query",
        value="" if is_file_search else user_query,
        height=100,
        key="refine_query_input",
        placeholder="Edit your query here and click \u2019Re-search\u2019 to get new results"
        + ("  (original input was a file upload)" if is_file_search else ""),
        label_visibility="collapsed",
    )
    if is_file_search:
        st.caption(f"\U0001f4c4 Original input: **{user_query[2:]}**")

    rb1, rb2, _ = st.columns([1, 1, 3])
    with rb1:
        if st.button(
            "\U0001f504 Re-search",
            type="primary",
            use_container_width=True,
            disabled=not refine_text.strip(),
            help="Run a new search with the query above.",
        ):
            st.session_state["pending_search_text"] = refine_text.strip()
            st.session_state["results"] = None
            st.session_state["profile"] = None
            st.rerun()
    with rb2:
        has_oai = bool(_ss("openai_key") or settings.openai_api_key)
        if st.button(
            "\u2728 Rephrase with AI",
            use_container_width=True,
            disabled=not (has_oai and refine_text.strip()),
            help=(
                "Rewrite your query into a cleaner, more effective search description."
                if has_oai else
                "Add an OpenAI API key in the sidebar to enable AI rephrasing."
            ),
        ):
            with st.spinner("\u2728 Rephrasing\u2026"):
                try:
                    oai_key = _ss("openai_key") or settings.openai_api_key
                    rephrased = rephrase_query(refine_text, oai_key, settings.openai_model)
                    st.session_state["refine_query_input"] = rephrased
                    st.session_state["user_query"] = rephrased
                    st.rerun()
                except Exception as exc:
                    st.error(f"Rephrasing failed: {exc}")

    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────
    with st.expander("🔧 Filter results", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            min_score = st.slider("Minimum match score", 0, 100, 0, key="flt_score")
        with fc2:
            remote_only = st.checkbox("Remote only", key="flt_remote")
        with fc3:
            all_src = sorted({j.job.source for j in ranked_jobs})
            chosen_src = st.multiselect("Source", all_src, default=all_src, key="flt_src")
        with fc4:
            desired_loc = profile.desired_location if profile else ""
            loc_filter = st.checkbox(
                "Hide wrong-region jobs",
                value=bool(desired_loc),
                key="flt_location",
                help="Remove jobs whose location is incompatible with your desired location.",
            )

    def _loc_ok(rj: RankedJob) -> bool:
        """Return False when job is flagged as region-incompatible."""
        if not loc_filter or not desired_loc:
            return True
        delta, _ = score_location_match(
            desired_loc,
            rj.job.location,
            rj.job.remote,
            rj.job.description[:400] if rj.job.description else "",
        )
        return delta >= 0

    filtered = [
        j for j in ranked_jobs
        if j.match_score >= min_score
        and (not remote_only or j.job.remote)
        and j.job.source in chosen_src
        and _loc_ok(j)
    ]

    st.markdown(f"Showing **{len(filtered)}** of {len(ranked_jobs)} jobs")
    st.markdown("")

    if not filtered:
        st.info("No jobs match the current filters. Relax the filters above.")
        return

    for rank, rj in enumerate(filtered, start=1):
        render_job_card(rj, rank)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Banner ───────────────────────────────────────────────────
    st.markdown(
        """
<div class="banner">
  <h1>🔍 AutoJob Finder</h1>
  <p>Upload your CV or describe the job you want — the AI finds and ranks the best matches for you.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Sidebar (always visible) ─────────────────────────────────
    location, remote, k = render_sidebar()

    # ── Re-search triggered from results page ───────────────────
    pending_text = _ss("pending_search_text")
    if pending_text:
        st.session_state["pending_search_text"] = None
        parser, searcher, matcher = _build_services()
        run_search(
            parser, searcher, matcher,
            None, pending_text,
            location, remote, k,
        )
        return

    # ── Results page ─────────────────────────────────────────────
    if _ss("results") is not None:
        render_results(_ss("results"))
        return

    # ── Input page ───────────────────────────────────────────────
    tab_cv, tab_text = st.tabs(["📄 Upload CV / Resume", "✍️ Describe What You Want"])

    uploaded_file   = None
    description_text = ""

    with tab_cv:
        st.markdown("#### Upload your CV or resume")
        st.markdown(
            "We'll automatically extract your skills, experience, and preferences. "
            "Supports **PDF**, **Word (DOCX)**, and **plain text** files."
        )
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["pdf", "docx", "doc", "txt"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.success(
                f"✅ **{uploaded_file.name}** loaded "
                f"({uploaded_file.size / 1024:.1f} KB)"
            )

    with tab_text:
        st.markdown("#### Describe the job you're looking for")
        st.markdown(
            "Be as detailed as you like — include your skills, experience level, "
            "desired role, preferred stack, location, and salary expectations."
        )
        description_text = st.text_area(
            "Your description",
            height=280,
            placeholder=(
                "Example:\n\n"
                "I'm a senior Python developer with 6 years of experience in backend "
                "engineering and machine learning. I'm proficient in FastAPI, Django, "
                "PostgreSQL, Docker, and AWS. I have strong experience with PyTorch and "
                "scikit-learn for building production ML pipelines.\n\n"
                "I'm looking for a remote ML Engineer or Senior Backend Developer role "
                "at a product company or startup. Preferred salary: $130k–$160k/year."
            ),
            label_visibility="collapsed",
            key="draft_query",
        )

        # ── Rephrase with AI ──────────────────────────────────────
        has_oai = bool(_ss("openai_key") or settings.openai_api_key)
        rp_col, _ = st.columns([1, 3])
        with rp_col:
            rephrase_clicked = st.button(
                "\u2728 Rephrase with AI",
                disabled=not (has_oai and bool(description_text.strip())),
                help=(
                    "Let AI rewrite your description into a concise, effective job search "
                    "query for better matching results. Requires an OpenAI API key."
                    if has_oai else
                    "Add an OpenAI API key in the sidebar to enable AI rephrasing."
                ),
            )
        if rephrase_clicked and description_text.strip():
            with st.spinner("\u2728 Rephrasing your query\u2026"):
                try:
                    oai_key = _ss("openai_key") or settings.openai_api_key
                    rephrased = rephrase_query(description_text, oai_key, settings.openai_model)
                    st.session_state["draft_query"] = rephrased
                    st.rerun()
                except Exception as exc:
                    st.error(f"Rephrasing failed: {exc}")

    st.markdown("")
    _, btn_col, _ = st.columns([1.5, 3, 1.5])
    with btn_col:
        search_clicked = st.button(
            f"🚀 Find My Top {k} Jobs",
            type="primary",
            use_container_width=True,
        )

    if search_clicked:
        has_input = bool(uploaded_file) or bool(description_text.strip())
        if not has_input:
            st.error("Please upload a CV **or** enter a job description before searching.")
            return

        parser, searcher, matcher = _build_services()

        if not searcher.available_sources:
            st.error(
                "No job sources are configured. "
                "Please add at least one API key in the sidebar (or leave blank to use the free Remotive source)."
            )
            return

        run_search(
            parser, searcher, matcher,
            uploaded_file, description_text,
            location, remote, k,
        )


if __name__ == "__main__":
    main()
