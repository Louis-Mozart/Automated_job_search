"""
Job search aggregator.

Free sources (no API key required)
────────────────────────────────────────────────────
Source              Coverage
────────────────────────────────────────────────────
WeWorkRemotely      100 recent remote jobs (RSS)  ← primary free source
HNHiring            300+ jobs from the monthly
                    "Ask HN: Who is hiring?"      ← great for AI/ML/research
Remotive            ~18 remote jobs (browse all)  ← small but simple
────────────────────────────────────────────────────

Paid sources (API key required)
────────────────────────────────────────────────────
JSearch / RapidAPI  LinkedIn · Indeed · Glassdoor · ZipRecruiter …
Adzuna              US, UK, AU, DE, FR, CA, NL, SG
────────────────────────────────────────────────────

Design principle
────────────────────────────────────────────────────
Free sources fetch the entire available pool (no server-side search filtering
because those filters are either broken or too coarse). The semantic matcher
in job_matcher.py then ranks the full pool and returns the top-k.
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from .models import Job, UserProfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _uid(source: str, title: str, company: str) -> str:
    raw = f"{source}|{title.lower().strip()}|{company.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:14]


def _get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 14,
) -> dict | list | None:
    """GET with error handling; returns parsed JSON or None."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("HTTP request failed [%s]: %s", url, exc)
        return None


_HTML_RE = re.compile(r"<[^>]+>")
_MULTISPACES = re.compile(r"\s{2,}")


def _strip_html(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#x2F;", "/").replace("&#x27;", "'"))
    return _MULTISPACES.sub(" ", text).strip()


# ─────────────────────────────────────────────────────────────────
# We Work Remotely  (RSS – free, ~100 jobs always fresh)
# ─────────────────────────────────────────────────────────────────

class WeWorkRemotelySearcher:
    """
    Parses the We Work Remotely RSS feed.
    Returns up to ~100 recent remote jobs with no API key required.
    """

    RSS_URL = "https://weworkremotely.com/remote-jobs.rss"
    HEADERS = {"User-Agent": "AutoJobFinder/1.0 (github.com/autojobfinder)"}

    def search(self, query: str, limit: int = 100) -> List[Job]:
        try:
            resp = requests.get(self.RSS_URL, headers=self.HEADERS, timeout=14)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logger.warning("WeWorkRemotely RSS failed: %s", exc)
            return []

        jobs: List[Job] = []
        for item in root.findall(".//item"):
            try:
                title_el  = item.find("title")
                link_el   = item.find("link")
                desc_el   = item.find("description")
                region_el = item.find("region")
                type_el   = item.find("type")
                pub_el    = item.find("pubDate")
                skills_el = item.find("skills")

                raw_title = (title_el.text or "") if title_el is not None else ""
                # RSS format: "Company: Job Title"
                if ":" in raw_title:
                    company, job_title = raw_title.split(":", 1)
                else:
                    company, job_title = "", raw_title

                description = _strip_html(desc_el.text or "") if desc_el is not None else ""
                skills: List[str] = []
                if skills_el is not None and skills_el.text:
                    skills = [s.strip() for s in skills_el.text.split(",") if s.strip()]

                jobs.append(
                    Job(
                        id=_uid("wwr", job_title.strip(), company.strip()),
                        title=job_title.strip(),
                        company=company.strip(),
                        location=(region_el.text or "Worldwide") if region_el is not None else "Worldwide",
                        remote=True,
                        job_type=((type_el.text or "").lower()) if type_el is not None else "full-time",
                        description=description,
                        url=link_el.text.strip() if link_el is not None and link_el.text else "",
                        posted_date=(pub_el.text or "") if pub_el is not None else "",
                        source="WeWorkRemotely",
                        required_skills=skills,
                    )
                )
            except Exception as exc:
                logger.debug("WWR item parse error: %s", exc)

        logger.info("WeWorkRemotely: fetched %d jobs.", len(jobs))
        return jobs[:limit]


# ─────────────────────────────────────────────────────────────────
# HN "Who is Hiring?"  (Algolia API – free, 300+ jobs/month)
# ─────────────────────────────────────────────────────────────────

_HN_SEARCH_URL   = "https://hn.algolia.com/api/v1/search_by_date"
_HN_COMMENTS_URL = "https://hn.algolia.com/api/v1/search"
_URL_RE = re.compile(r'href=["\']?(https?://[^\s"\'<>]+)', re.IGNORECASE)


def _find_url_in_text(text: str) -> str:
    m = _URL_RE.search(text)
    if m:
        return m.group(1)
    m2 = re.search(r"https?://\S+", text)
    return m2.group(0).rstrip(".,)") if m2 else ""


def _parse_hn_first_line(comment_html: str) -> tuple[str, str, str]:
    """
    Extract (title, company, location) from HN "Who is Hiring?" comment.
    Typical format: "Company | City, State | REMOTE | FULLTIME\\nDescription…"
    """
    text = _strip_html(comment_html)
    first_line = text.split("\n")[0][:280]
    parts = [p.strip() for p in first_line.split("|")]

    if len(parts) >= 2:
        title   = parts[0][:120]
        company = parts[1][:100] if len(parts) > 1 else ""
    else:
        title   = first_line[:120]
        company = ""

    location_kws = ["remote", "onsite", "hybrid", "usa", "eu", "uk", "london",
                    "berlin", "paris", "nyc", "sf", "boston", "toronto", "anywhere"]
    location = ""
    for part in parts[2:6]:
        if any(kw in part.lower() for kw in location_kws):
            location = part
            break

    if not location and "remote" in text.lower():
        location = "Remote"

    return title.strip(), company.strip(), location.strip()


class HNHiringSearcher:
    """
    Parses the monthly "Ask HN: Who is hiring?" thread via the Algolia HN API.
    Excellent source for AI, ML, research, and deep-tech roles.
    """

    MAX_PAGES = 6   # 100 comments/page × 6 = up to 600 comments

    def _get_latest_post_id(self) -> Optional[str]:
        data = _get(
            _HN_SEARCH_URL,
            params={"query": "Ask HN: Who is hiring", "tags": "story", "hitsPerPage": 1},
        )
        hits = (data or {}).get("hits", [])
        return hits[0]["objectID"] if hits else None

    def _fetch_comments(self, post_id: str, limit: int) -> List[dict]:
        comments: List[dict] = []
        page = 0
        while len(comments) < limit:
            data = _get(
                _HN_COMMENTS_URL,
                params={"tags": f"comment,story_{post_id}", "hitsPerPage": 100, "page": page},
            )
            if not data:
                break
            batch = data.get("hits", [])
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
            if page >= self.MAX_PAGES:
                break
        return comments[:limit]

    def search(self, query: str, limit: int = 500) -> List[Job]:
        post_id = self._get_latest_post_id()
        if not post_id:
            logger.warning("HNHiring: could not find latest 'Who is hiring?' post.")
            return []

        logger.info("HNHiring: post_id=%s, fetching up to %d comments …", post_id, limit)
        comments = self._fetch_comments(post_id, limit)
        logger.info("HNHiring: %d raw comments fetched.", len(comments))

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        jobs: List[Job] = []
        for c in comments:
            raw_html = c.get("comment_text") or ""
            if not raw_html.strip() or len(raw_html) < 80:
                continue

            plain = _strip_html(raw_html)
            plain_lower = plain.lower()

            # Keep comment if it matches a query keyword OR looks like a job post
            has_keyword    = bool(query_words & set(re.findall(r"\b\w{3,}\b", plain_lower)))
            has_job_signal = any(
                kw in plain_lower
                for kw in ["full-time", "fulltime", "full time", "hiring", "engineer",
                           "developer", "researcher", "scientist", "remote", "onsite",
                           "position", "role", "salary", "equity"]
            )
            if not (has_keyword or has_job_signal):
                continue

            title, company, location = _parse_hn_first_line(raw_html)
            url = (_find_url_in_text(raw_html)
                   or f"https://news.ycombinator.com/item?id={c.get('objectID','')}")

            jobs.append(
                Job(
                    id=_uid("hn", title, company or c.get("objectID", "")),
                    title=title or "Job opening (see description)",
                    company=company or "via HN Hiring",
                    location=location or "See description",
                    remote="remote" in plain_lower,
                    description=plain,
                    url=url,
                    posted_date=c.get("created_at", ""),
                    source="HN Hiring",
                )
            )

        logger.info("HNHiring: %d job-like comments returned.", len(jobs))
        return jobs


# ─────────────────────────────────────────────────────────────────
# Remotive  (browse all categories – keyword search is broken)
# ─────────────────────────────────────────────────────────────────

_REMOTIVE_CATEGORIES = [
    "software-development", "ai-ml", "data", "devops",
    "product", "all-others",
]


class RemotiveSearcher:
    """
    Browses all Remotive categories and returns every available job.
    NOTE: Remotive's keyword search returns 0 results (broken endpoint).
    """

    URL = "https://remotive.com/api/remote-jobs"

    def search(self, query: str, limit: int = 100) -> List[Job]:
        jobs: List[Job] = []
        seen: set[str] = set()

        for cat in _REMOTIVE_CATEGORIES:
            data = _get(self.URL, params={"category": cat})
            if not data:
                continue
            for item in (data.get("jobs") or []):
                uid = _uid("remotive", item.get("title", ""), item.get("company_name", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                jobs.append(
                    Job(
                        id=uid,
                        title=item.get("title", ""),
                        company=item.get("company_name", ""),
                        location=item.get("candidate_required_location") or "Remote",
                        description=_strip_html(item.get("description", "")),
                        url=item.get("url", ""),
                        posted_date=item.get("publication_date", ""),
                        job_type=(item.get("job_type") or "").lower(),
                        remote=True,
                        source="Remotive",
                        required_skills=item.get("tags") or [],
                    )
                )
        logger.info("Remotive: fetched %d jobs.", len(jobs))
        return jobs[:limit]


# ─────────────────────────────────────────────────────────────────
# JSearch via RapidAPI  (free tier: 200 req / month)
# ─────────────────────────────────────────────────────────────────

class JSearchSearcher:
    """Aggregates LinkedIn, Indeed, Glassdoor, ZipRecruiter via RapidAPI."""

    URL = "https://jsearch.p.rapidapi.com/search"

    def __init__(self, api_key: str) -> None:
        self._headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        }

    def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: bool = False,
        limit: int = 30,
    ) -> List[Job]:
        search_str = query
        if location:
            search_str += f" in {location}"
        if remote:
            search_str += " remote"

        jobs: List[Job] = []
        page = 1
        while len(jobs) < limit:
            data = _get(
                self.URL,
                params={"query": search_str, "page": str(page), "num_pages": "1", "date_posted": "all"},
                headers=self._headers,
            )
            if not data:
                break
            items = data.get("data") or []
            if not items:
                break
            for item in items:
                sal_min = item.get("job_min_salary")
                sal_max = item.get("job_max_salary")
                city    = item.get("job_city") or ""
                state   = item.get("job_state") or ""
                country = item.get("job_country") or ""
                loc     = ", ".join(p for p in [city, state, country] if p)
                jobs.append(
                    Job(
                        id=item.get("job_id") or _uid("jsearch", item.get("job_title", ""), item.get("employer_name", "")),
                        title=item.get("job_title", ""),
                        company=item.get("employer_name", ""),
                        location=loc or None,
                        description=item.get("job_description", ""),
                        url=item.get("job_apply_link") or item.get("job_google_link", ""),
                        salary_min=float(sal_min) if sal_min else None,
                        salary_max=float(sal_max) if sal_max else None,
                        salary_currency=item.get("job_salary_currency") or "USD",
                        salary_period=(item.get("job_salary_period") or "year").lower(),
                        posted_date=item.get("job_posted_at_datetime_utc", ""),
                        job_type=(item.get("job_employment_type") or "").lower(),
                        remote=bool(item.get("job_is_remote", False)),
                        source="JSearch",
                        required_skills=item.get("job_required_skills") or [],
                        raw_data=item,
                    )
                )
                if len(jobs) >= limit:
                    break
            page += 1
            if page > 5:
                break
        return jobs


# ─────────────────────────────────────────────────────────────────
# Adzuna  (free tier: 250 req / day)
# ─────────────────────────────────────────────────────────────────

class AdzunaSearcher:
    BASE = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

    def __init__(self, app_id: str, app_key: str, country: str = "us") -> None:
        self._app_id  = app_id
        self._app_key = app_key
        self._country = country

    def search(self, query: str, location: Optional[str] = None, limit: int = 30) -> List[Job]:
        jobs: List[Job] = []
        page = 1
        per_page = min(50, limit)
        while len(jobs) < limit:
            params: dict = {
                "app_id": self._app_id, "app_key": self._app_key,
                "what": query, "results_per_page": per_page,
                "content-type": "application/json",
            }
            if location:
                params["where"] = location
            data = _get(self.BASE.format(country=self._country, page=page), params=params)
            if not data:
                break
            results = data.get("results") or []
            if not results:
                break
            for item in results:
                sal_min = item.get("salary_min")
                sal_max = item.get("salary_max")
                jobs.append(
                    Job(
                        id=_uid("adzuna", item.get("title", ""), item.get("company", {}).get("display_name", "")),
                        title=item.get("title", ""),
                        company=item.get("company", {}).get("display_name", ""),
                        location=item.get("location", {}).get("display_name"),
                        description=item.get("description", ""),
                        url=item.get("redirect_url", ""),
                        salary_min=float(sal_min) if sal_min else None,
                        salary_max=float(sal_max) if sal_max else None,
                        salary_currency="USD",
                        salary_period="year",
                        posted_date=item.get("created", ""),
                        job_type=(item.get("contract_time") or "").replace("_", "-"),
                        source="Adzuna",
                    )
                )
                if len(jobs) >= limit:
                    break
            page += 1
            if page > 5:
                break
        return jobs


# ─────────────────────────────────────────────────────────────────
# Bundesagentur für Arbeit  (German Federal Employment Agency)
# Free public API – no registration needed. Germany / DACH region.
# All job types: tech AND non-tech (waiter, cleaner, teacher …)
# ─────────────────────────────────────────────────────────────────

class BundesagenturSearcher:
    """
    German Federal Employment Agency – completely free, no API key required.
    Covers ALL job types in Germany / Austria / Switzerland.
    Results are strongest for German-language queries and German cities.
    """

    BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
    JOB_URL  = "https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
    HEADERS  = {"X-API-Key": "jobboerse-jobsuche", "User-Agent": "AutoJobFinder/1.0"}

    # Common EN → DE translations so queries like "waiter" hit the right results
    _EN_TO_DE: dict[str, str] = {
        "waiter": "Kellner", "waitress": "Kellnerin", "server": "Servicekraft",
        "bartender": "Barkeeper", "barista": "Barista",
        "chef": "Koch", "cook": "Koch", "kitchen staff": "Küchenhilfe",
        "cleaner": "Reinigungskraft", "cleaning": "Reinigung",
        "car cleaner": "Reinigung", "car wash": "Fahrzeugpflege",
        "housekeeper": "Hauswirtschaft", "housekeeping": "Hauswirtschaft",
        "janitor": "Hausmeister", "maintenance": "Instandhaltung",
        "teacher": "Lehrer", "tutor": "Nachhilfelehrer",
        "childcare": "Kinderpflege", "kindergarten teacher": "Erzieherin",
        "driver": "Fahrer", "delivery driver": "Lieferfahrer",
        "warehouse": "Lagermitarbeiter", "logistics": "Logistik",
        "cashier": "Kassierer", "retail": "Einzelhandel",
        "sales assistant": "Verkäufer", "shop assistant": "Verkäufer",
        "security guard": "Sicherheitsmitarbeiter", "security": "Sicherheitsdienst",
        "gardener": "Gärtner", "landscaper": "Landschaftsgärtner",
        "electrician": "Elektriker", "plumber": "Klempner",
        "carpenter": "Tischler", "painter": "Maler",
        "mechanic": "Mechaniker", "welder": "Schweißer",
        "caregiver": "Pflegekraft", "care worker": "Pflegehelfer",
        "receptionist": "Empfangsmitarbeiter",
    }

    # EN → German spelling of city names (BA only accepts German names)
    _CITY_DE: dict[str, str] = {
        "munich": "München", "cologne": "Köln", "nuremberg": "Nürnberg",
        "dusseldorf": "Düsseldorf", "düsseldorf": "Düsseldorf",
        "vienna": "Wien", "zurich": "Zürich", "zurich": "Zürich",
    }

    # Countries whose cities BA actually covers
    _DACH_COUNTRIES = {"germany", "austria", "switzerland"}

    def _normalize_location(self, loc: str) -> tuple[str, bool]:
        """
        Return (ba_location, is_dach).
        - ba_location: German spelling the BA API understands
        - is_dach: False → skip BA entirely (non-German location)
        """
        if not loc:
            return "", True   # no constraint = search Germany-wide

        from .location_filter import _CITY_TO_COUNTRY
        loc_lower = loc.lower().strip()

        # City lookup (longest match first)
        for city in sorted(_CITY_TO_COUNTRY, key=len, reverse=True):
            if city in loc_lower:
                country = _CITY_TO_COUNTRY[city]
                if country in self._DACH_COUNTRIES:
                    # Use German spelling if available, else original
                    return self._CITY_DE.get(city, loc), True
                return loc, False   # definitely non-DACH

        # Fall through: unknown location – try as-is, BA returns 0 if not found
        return self._CITY_DE.get(loc_lower, loc), True

    def _queries_for(self, query: str) -> list[str]:
        """Return [original, german_translation] if a translation exists."""
        q_lower = query.lower().strip()
        queries = [query]
        # Longest-match first so "car cleaner" beats "cleaner"
        for en in sorted(self._EN_TO_DE, key=len, reverse=True):
            if en in q_lower:
                de = self._EN_TO_DE[en]
                if de.lower() != q_lower:
                    queries.append(de)
                break
        return queries

    def search(self, query: str, location: str = "", limit: int = 100) -> List[Job]:
        ba_location, is_dach = self._normalize_location(location)
        if not is_dach:
            logger.info("Bundesagentur: skipping – location '%s' is outside DACH.", location)
            return []

        all_jobs: List[Job] = []
        seen: set[str] = set()

        for q in self._queries_for(query):
            params: dict = {
                "was": q,
                "page": 1,
                "size": min(limit, 100),
            }
            if ba_location:
                params["wo"] = ba_location
            try:
                resp = requests.get(
                    self.BASE_URL, params=params,
                    headers=self.HEADERS, timeout=14,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Bundesagentur search failed for '%s': %s", q, exc)
                continue

            for item in data.get("stellenangebote") or []:
                try:
                    refnr   = item.get("refnr", "")
                    if refnr in seen:
                        continue
                    seen.add(refnr)

                    titel   = item.get("titel", "")
                    beruf   = item.get("beruf", "")       # occupation category
                    company = item.get("arbeitgeber", "")
                    loc     = item.get("arbeitsort", {})
                    city    = loc.get("ort", "")
                    region  = loc.get("region", "")
                    country = loc.get("land", "Deutschland")
                    location_str = ", ".join(filter(None, [city, region, country]))

                    # Construct a pseudo-description from available metadata
                    # (the list endpoint does not include full text)
                    desc_parts = []
                    if beruf:
                        desc_parts.append(f"Occupation: {beruf}")
                    desc_parts.append(f"Title: {titel}")
                    if company:
                        desc_parts.append(f"Employer: {company}")
                    if city:
                        desc_parts.append(f"Location: {location_str}")
                    description = ". ".join(desc_parts)

                    all_jobs.append(Job(
                        id=_uid("ba", titel, company),
                        title=titel,
                        company=company,
                        location=location_str,
                        remote=False,
                        description=description,
                        url=self.JOB_URL.format(refnr=refnr),
                        posted_date=item.get("aktuelleVeroeffentlichungsdatum", ""),
                        source="Bundesagentur",
                    ))
                except Exception as exc:
                    logger.debug("BA item parse error: %s", exc)

            if len(all_jobs) >= limit:
                break

        logger.info("Bundesagentur: %d jobs for '%s' in '%s' (ba_loc='%s').",
                    len(all_jobs), query, location, ba_location)
        return all_jobs[:limit]


# ─────────────────────────────────────────────────────────────────
# Jooble  (worldwide – free API key, all job types)
# Register for free at jooble.org/api/index
# ─────────────────────────────────────────────────────────────────

class JoobleSearcher:
    """
    Jooble – worldwide job aggregator (tech and non-tech).
    Free API key: register at https://jooble.org/api/index
    500 free searches / month.
    """

    BASE_URL = "https://jooble.org/api/{api_key}"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._url = self.BASE_URL.format(api_key=api_key)

    def search(
        self,
        query: str,
        location: str = "",
        limit: int = 20,
    ) -> List[Job]:
        body: dict = {"keywords": query}
        if location:
            body["location"] = location
        try:
            resp = requests.post(self._url, json=body, timeout=14)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Jooble search failed: %s", exc)
            return []

        jobs: List[Job] = []
        for item in (data.get("jobs") or [])[:limit]:
            try:
                title   = item.get("title", "")
                company = item.get("company", "")
                loc_str = item.get("location", "")
                snippet = _strip_html(item.get("snippet", ""))
                link    = item.get("link", "")
                jtype   = item.get("type", "")
                updated = item.get("updated", "")[:10]
                remote  = "remote" in (loc_str + " " + snippet).lower()

                jobs.append(Job(
                    id=_uid("jooble", title, company),
                    title=title,
                    company=company,
                    location=loc_str,
                    remote=remote,
                    description=snippet,
                    url=link,
                    job_type=jtype,
                    posted_date=updated,
                    source="Jooble",
                ))
            except Exception as exc:
                logger.debug("Jooble item parse error: %s", exc)

        logger.info("Jooble: %d jobs for '%s'.", len(jobs), query)
        return jobs


# ─────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────

class JobSearcher:
    """
    Orchestrates multiple sources and returns a deduplicated ``List[Job]``
    ready to be ranked by :class:`JobMatcher`.

    Always active (no key required):
      - WeWorkRemotely      (~100 remote tech jobs via RSS)
      - HNHiring            (300+ jobs, great for AI/ML/research)
      - Remotive            (~18 remote jobs, browse all categories)
      - Bundesagentur (BA)  (all job types in Germany/DACH – waiter, cleaner…)

    Optional (unlocked by API keys – free registration):
      - Jooble              (worldwide, ALL job types, 500 req/month free)

    Optional (unlocked by API keys – paid / limited free tier):
      - JSearch / RapidAPI  (LinkedIn · Indeed · Glassdoor · ZipRecruiter)
      - Adzuna              (250 free req/day, many countries)
    """

    def __init__(
        self,
        rapidapi_key: Optional[str] = None,
        adzuna_app_id: Optional[str] = None,
        adzuna_app_key: Optional[str] = None,
        adzuna_country: str = "us",
        jooble_api_key: Optional[str] = None,
    ) -> None:
        self._sources: dict = {}

        if rapidapi_key:
            self._sources["JSearch"] = JSearchSearcher(rapidapi_key)
            logger.info("JSearch (RapidAPI) enabled.")

        if adzuna_app_id and adzuna_app_key:
            self._sources["Adzuna"] = AdzunaSearcher(adzuna_app_id, adzuna_app_key, adzuna_country)
            logger.info("Adzuna enabled (country=%s).", adzuna_country)

        if jooble_api_key:
            self._sources["Jooble"] = JoobleSearcher(jooble_api_key)
            logger.info("Jooble enabled (worldwide, all job types).")

        # Always-on free sources
        # Bundesagentur first: covers all job types in Germany (non-tech too).
        # Tech sources follow; BA results won't be crowded out at low max_results.
        self._sources["Bundesagentur"]  = BundesagenturSearcher()
        self._sources["WeWorkRemotely"] = WeWorkRemotelySearcher()
        self._sources["HNHiring"]       = HNHiringSearcher()
        self._sources["Remotive"]       = RemotiveSearcher()

        logger.info("Active sources: %s", ", ".join(self._sources))

    @property
    def available_sources(self) -> List[str]:
        return list(self._sources.keys())

    @property
    def has_paid_sources(self) -> bool:
        return any(k in self._sources for k in ("JSearch", "Adzuna", "Jooble"))

    def search(
        self,
        profile: UserProfile,
        location: Optional[str] = None,
        remote: bool = False,
        max_results: int = 600,
    ) -> List[Job]:
        """
        Multi-source search. Free sources fetch their full catalogue;
        paid sources use generated search queries. All results are
        deduplicated and returned for semantic ranking.
        """
        queries = list(profile.search_queries)
        if not queries:
            parts: List[str] = []
            if profile.current_title:
                parts.append(profile.current_title)
            elif profile.desired_titles:
                parts.append(profile.desired_titles[0])
            if profile.skills:
                parts.extend(profile.skills[:3])
            queries = [" ".join(parts)] if parts else ["software engineer"]

        primary_query = queries[0]

        all_jobs: List[Job] = []
        seen: set[str] = set()

        def _add(batch: List[Job]) -> None:
            for job in batch:
                if job.id not in seen:
                    seen.add(job.id)
                    all_jobs.append(job)

        for source_name, searcher in self._sources.items():
            try:
                if isinstance(searcher, WeWorkRemotelySearcher):
                    _add(searcher.search(primary_query, limit=200))

                elif isinstance(searcher, RemotiveSearcher):
                    _add(searcher.search(primary_query, limit=200))

                elif isinstance(searcher, HNHiringSearcher):
                    _add(searcher.search(primary_query, limit=600))

                elif isinstance(searcher, BundesagenturSearcher):
                    # Location-aware: pass user location; BA returns results
                    # only for German/DACH cities (0 results for Paris etc.)
                    _add(searcher.search(primary_query, location=location or "", limit=100))

                elif isinstance(searcher, JoobleSearcher):
                    per_q = max(10, 60 // max(1, len(queries[:3])))
                    for q in queries[:3]:
                        _add(searcher.search(q, location=location or "", limit=per_q))

                elif isinstance(searcher, JSearchSearcher):
                    per_q = max(20, 120 // max(1, len(queries[:3])))
                    for q in queries[:3]:
                        _add(searcher.search(q, location=location, remote=remote, limit=per_q))

                elif isinstance(searcher, AdzunaSearcher):
                    per_q = max(20, 120 // max(1, len(queries[:3])))
                    for q in queries[:3]:
                        _add(searcher.search(q, location=location, limit=per_q))

            except Exception as exc:
                logger.error("Error in source '%s': %s", source_name, exc)

        logger.info(
            "Total unique jobs: %d from %d source(s).", len(all_jobs), len(self._sources)
        )
        return all_jobs[:max_results]
