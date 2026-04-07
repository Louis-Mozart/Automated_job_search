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
        # The monthly thread is always posted by the 'whoishiring' account.
        # Adding author_whoishiring prevents matching unrelated Show HN / Ask HN posts.
        data = _get(
            _HN_SEARCH_URL,
            params={
                "query": "Ask HN: Who is hiring",
                "tags": "story,author_whoishiring",
                "hitsPerPage": 1,
            },
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

# City / region → ISO-2 country code used to drop clearly-wrong-country results
_LOCATION_TO_COUNTRY_CODE: dict[str, str] = {
    # France
    "paris": "FR", "lyon": "FR", "marseille": "FR", "toulouse": "FR",
    "nice": "FR", "nantes": "FR", "strasbourg": "FR", "montpellier": "FR",
    "bordeaux": "FR", "lille": "FR", "rennes": "FR", "grenoble": "FR",
    "france": "FR",
    # UK
    "london": "GB", "manchester": "GB", "birmingham": "GB", "leeds": "GB",
    "glasgow": "GB", "edinburgh": "GB", "bristol": "GB", "united kingdom": "GB",
    # Germany
    "berlin": "DE", "munich": "DE", "hamburg": "DE", "frankfurt": "DE",
    "cologne": "DE", "düsseldorf": "DE", "dusseldorf": "DE", "germany": "DE",
    # Spain
    "madrid": "ES", "barcelona": "ES", "seville": "ES", "valencia": "ES", "spain": "ES",
    # Italy
    "rome": "IT", "milan": "IT", "naples": "IT", "turin": "IT", "italy": "IT",
    # Netherlands / Belgium
    "amsterdam": "NL", "rotterdam": "NL", "the hague": "NL", "netherlands": "NL",
    "brussels": "BE", "antwerp": "BE", "ghent": "BE", "belgium": "BE",
    # Switzerland / Austria
    "zurich": "CH", "geneva": "CH", "bern": "CH", "switzerland": "CH",
    "vienna": "AT", "graz": "AT", "austria": "AT",
    # Nordics
    "stockholm": "SE", "gothenburg": "SE", "sweden": "SE",
    "oslo": "NO", "bergen": "NO", "norway": "NO",
    "copenhagen": "DK", "denmark": "DK",
    "helsinki": "FI", "finland": "FI",
    # Portugal / Eastern Europe
    "lisbon": "PT", "porto": "PT", "portugal": "PT",
    "warsaw": "PL", "krakow": "PL", "poland": "PL",
    "prague": "CZ", "czech": "CZ",
    "budapest": "HU", "hungary": "HU",
    "bucharest": "RO", "romania": "RO",
    # North America
    "new york": "US", "los angeles": "US", "chicago": "US", "houston": "US",
    "san francisco": "US", "seattle": "US", "boston": "US", "miami": "US",
    "austin": "US", "denver": "US", "united states": "US", "usa": "US",
    "toronto": "CA", "montreal": "CA", "vancouver": "CA", "canada": "CA",
    # APAC
    "sydney": "AU", "melbourne": "AU", "brisbane": "AU", "australia": "AU",
    "tokyo": "JP", "osaka": "JP", "japan": "JP",
    "singapore": "SG",
    "delhi": "IN", "mumbai": "IN", "bangalore": "IN", "india": "IN",
    "beijing": "CN", "shanghai": "CN", "china": "CN",
    # Middle East / Other
    "dubai": "AE", "abu dhabi": "AE",
    "são paulo": "BR", "rio de janeiro": "BR", "brazil": "BR",
}


def _country_code_for_location(location: str) -> Optional[str]:
    """
    Return the ISO-2 country code for a free-text location string, or None if unknown.
    Uses longest-match so 'new york' beats 'york'.
    """
    loc_lower = location.lower()
    for key in sorted(_LOCATION_TO_COUNTRY_CODE, key=len, reverse=True):
        if key in loc_lower:
            return _LOCATION_TO_COUNTRY_CODE[key]
    return None


# Adzuna country codes supported by the API (lowercase ISO-2)
_ADZUNA_SUPPORTED: set[str] = {
    "us", "gb", "de", "fr", "au", "ca", "nl", "sg",
    "at", "be", "br", "in", "it", "mx", "nz", "pl", "ru", "za",
}


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

        # Determine the expected country so we can drop clearly-wrong results.
        # E.g. searching "cooker in Paris" should not return New York cooker jobs.
        expected_country: Optional[str] = (
            _country_code_for_location(location) if location else None
        )

        jobs: List[Job] = []
        _tried_fallback = False
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

            # JSearch has very sparse coverage outside the US/English-speaking world.
            # If the location-scoped query returns nothing, retry with the bare query
            # and rely solely on the country post-filter to keep results relevant.
            if not items and location and not _tried_fallback:
                _tried_fallback = True
                search_str = query  # drop "in <location>" suffix
                if remote:
                    search_str += " remote"
                logger.debug("JSearch: 0 results for location-scoped query, retrying as '%s'", search_str)
                continue

            if not items:
                break
            for item in items:
                sal_min = item.get("job_min_salary")
                sal_max = item.get("job_max_salary")
                city    = item.get("job_city") or ""
                state   = item.get("job_state") or ""
                country = item.get("job_country") or ""
                loc     = ", ".join(p for p in [city, state, country] if p)
                is_remote = bool(item.get("job_is_remote", False))

                # Drop jobs in a different country (or unknown country) when we
                # expect a specific one. The 'country and' guard was removed so that
                # jobs with an empty country field don't slip through the filter.
                if expected_country and not is_remote:
                    if not country or country.upper() != expected_country.upper():
                        logger.debug(
                            "JSearch: dropping '%s' – country %r ≠ expected %s",
                            item.get("job_title", ""), country, expected_country,
                        )
                        continue

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
                        remote=is_remote,
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
        # Auto-detect the Adzuna country endpoint from the location string so that
        # typing "Paris" automatically hits the French database (/fr/search/) instead
        # of the default US database. Falls back to the configured country if the
        # location is unknown or not covered by Adzuna.
        country = self._country
        if location:
            iso = _country_code_for_location(location)
            if iso and iso.lower() in _ADZUNA_SUPPORTED:
                country = iso.lower()
                logger.debug("Adzuna: auto-selected country '%s' from location '%s'.", country, location)

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
            data = _get(self.BASE.format(country=country, page=page), params=params)
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
# France Travail  (formerly Pôle Emploi)
# Free – register once at francetravail.io/partenaire to get credentials.
# Covers ALL job types in France (tech AND non-tech).
# ─────────────────────────────────────────────────────────────────

class FranceTravailSearcher:
    """
    France's official job board – completely free with a registered API key.
    Register at https://francetravail.io/partenaire (takes ~5 minutes).
    Covers all job types: cook, waiter, cleaner, developer, nurse …
    Results are strongest for French-language queries and French cities.
    """

    TOKEN_URL  = "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
    SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"

    # EN → FR keyword translations for better matching
    _EN_TO_FR: dict[str, str] = {
        "cook": "cuisinier", "cooker": "cuisinier", "cooking": "cuisine",
        "chef": "chef cuisinier", "kitchen staff": "aide cuisine",
        "waiter": "serveur", "waitress": "serveuse", "server": "serveur",
        "bartender": "barman", "barista": "barista",
        "cleaner": "agent entretien", "cleaning": "nettoyage",
        "car cleaner": "nettoyage véhicule", "car wash": "lavage auto",
        "housekeeper": "femme de chambre", "housekeeping": "gouvernante",
        "janitor": "gardien", "maintenance": "maintenance",
        "teacher": "professeur", "tutor": "tuteur",
        "childcare": "auxiliaire puériculture", "kindergarten teacher": "éducateur jeunes enfants",
        "driver": "chauffeur", "delivery driver": "livreur",
        "warehouse": "magasinier", "logistics": "logistique",
        "cashier": "caissier", "retail": "vendeur",
        "sales assistant": "vendeur", "shop assistant": "vendeur",
        "security guard": "agent sécurité", "security": "sécurité",
        "gardener": "jardinier", "landscaper": "paysagiste",
        "electrician": "électricien", "plumber": "plombier",
        "carpenter": "menuisier", "painter": "peintre",
        "mechanic": "mécanicien", "welder": "soudeur",
        "caregiver": "aide soignant", "care worker": "aide soignant", "nurse": "infirmier",
        "receptionist": "réceptionniste",
        "developer": "développeur", "python developer": "développeur python",
        "engineer": "ingénieur", "data scientist": "data scientist",
        "project manager": "chef de projet",
    }

    # City → INSEE département code for spatial filtering
    _CITY_TO_DEPT: dict[str, str] = {
        "paris": "75", "hauts-de-seine": "92", "seine-saint-denis": "93",
        "val-de-marne": "94", "île-de-france": "75",
        "lyon": "69", "villeurbanne": "69",
        "marseille": "13", "aix-en-provence": "13",
        "toulouse": "31",
        "nice": "06", "cannes": "06", "antibes": "06",
        "nantes": "44",
        "strasbourg": "67",
        "montpellier": "34",
        "bordeaux": "33",
        "lille": "59", "roubaix": "59", "tourcoing": "59",
        "rennes": "35",
        "grenoble": "38",
        "nancy": "54",
        "tours": "37",
        "metz": "57",
        "clermont-ferrand": "63",
        "dijon": "21",
        "rouen": "76",
        "reims": "51",
        "saint-étienne": "42",
        "toulon": "83",
    }

    # France-related location keywords
    _FRANCE_KEYWORDS = {
        "paris", "france", "française", "lyon", "marseille", "toulouse",
        "nice", "nantes", "strasbourg", "montpellier", "bordeaux", "lille",
        "rennes", "grenoble", "nancy", "metz", "rouen", "reims", "dijon",
        "toulon", "saint-étienne", "clermont-ferrand",
    }

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id     = client_id
        self._client_secret = client_secret
        self._token: Optional[str] = None
        self._token_expiry: float  = 0.0

    def _get_token(self) -> Optional[str]:
        import time
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        try:
            resp = requests.post(
                self.TOKEN_URL,
                params={"realm": "/partenaire"},
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                    "scope":         "api_offresdemploiv2 o2dsoffre",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token        = data["access_token"]
            self._token_expiry = time.time() + data.get("expires_in", 1800)
            return self._token
        except Exception as exc:
            logger.warning("France Travail: token request failed: %s", exc)
            return None

    def _is_france_location(self, loc: str) -> bool:
        """Return True if the location string refers to a French city/region."""
        if not loc:
            return True  # no constraint → search France-wide
        loc_lower = loc.lower()
        return any(kw in loc_lower for kw in self._FRANCE_KEYWORDS)

    def _translate_query(self, query: str) -> str:
        """Translate the first matching EN term to French, or return query unchanged."""
        q_lower = query.lower().strip()
        for en in sorted(self._EN_TO_FR, key=len, reverse=True):
            if en in q_lower:
                return self._EN_TO_FR[en]
        return query

    def search(self, query: str, location: str = "", limit: int = 100) -> List[Job]:
        if not self._is_france_location(location):
            logger.info("France Travail: skipping – location '%s' is outside France.", location)
            return []

        token = self._get_token()
        if not token:
            return []

        # Detect département for spatial filtering
        dept: Optional[str] = None
        loc_lower = location.lower()
        for city, dept_code in sorted(self._CITY_TO_DEPT.items(), key=lambda x: len(x[0]), reverse=True):
            if city in loc_lower:
                dept = dept_code
                break

        fr_query = self._translate_query(query)
        params: dict = {
            "motsCles":         fr_query,
            "nbreMaxFrParPage": min(limit, 150),
        }
        if dept:
            params["departement"] = dept

        try:
            resp = requests.get(
                self.SEARCH_URL,
                params=params,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=14,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("France Travail search failed for '%s': %s", fr_query, exc)
            return []

        jobs: List[Job] = []
        for item in (data.get("resultats") or [])[:limit]:
            try:
                jid     = item.get("id", "")
                title   = item.get("intitule", "")
                company = (item.get("entreprise") or {}).get("nom", "")
                lieu    = item.get("lieuTravail") or {}
                loc_str = lieu.get("libelle", location)
                url     = (
                    item.get("origineOffre", {}).get("urlOrigine")
                    or f"https://candidat.francetravail.fr/offres/recherche/detail/{jid}"
                )
                desc     = item.get("description", "")
                contract = item.get("typeContratLibelle", "")
                date_pub = item.get("dateCreation", "")[:10]

                jobs.append(Job(
                    id=_uid("ft", title, company),
                    title=title,
                    company=company,
                    location=loc_str,
                    remote=False,
                    description=desc,
                    url=url,
                    posted_date=date_pub,
                    job_type=contract,
                    source="France Travail",
                ))
            except Exception as exc:
                logger.debug("France Travail item parse error: %s", exc)

        logger.info("France Travail: %d jobs for '%s' in '%s' (dept=%s).",
                    len(jobs), fr_query, location, dept)
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

    Optional (free registration):
      - France Travail      (all job types in France – cook, waiter, dev…)
      - Jooble              (worldwide, ALL job types, 500 req/month free)

    Optional (paid / limited free tier):
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
        france_travail_client_id: Optional[str] = None,
        france_travail_client_secret: Optional[str] = None,
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

        if france_travail_client_id and france_travail_client_secret:
            self._sources["FranceTravail"] = FranceTravailSearcher(
                france_travail_client_id, france_travail_client_secret
            )
            logger.info("France Travail enabled (all job types in France).")

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

                elif isinstance(searcher, FranceTravailSearcher):
                    # Location-aware: pass user location; FT returns results
                    # only for French cities (skips non-France locations).
                    per_q = max(20, 100 // max(1, len(queries[:3])))
                    for q in queries[:3]:
                        _add(searcher.search(q, location=location or "", limit=per_q))

                elif isinstance(searcher, JoobleSearcher):
                    per_q = max(10, 60 // max(1, len(queries[:3])))
                    for q in queries[:3]:
                        _add(searcher.search(q, location=location or "", limit=per_q))

                elif isinstance(searcher, JSearchSearcher):
                    # JSearch index is overwhelmingly US / English-speaking world.
                    # For other regions it almost always returns 0 results and then
                    # falls back to unfiltered US jobs – so skip it entirely there.
                    _JSEARCH_STRONG = {"US", "GB", "CA", "AU", "SG", "IN", "NZ"}
                    jsearch_country = (
                        _country_code_for_location(location) if location else None
                    )
                    if jsearch_country and jsearch_country.upper() not in _JSEARCH_STRONG:
                        logger.info(
                            "JSearch: skipping – location '%s' (%s) is outside "
                            "JSearch's strong-coverage region.",
                            location, jsearch_country,
                        )
                    else:
                        per_q = max(20, 120 // max(1, len(queries[:3])))
                        for q in queries[:3]:
                            _add(searcher.search(q, location=location, remote=remote, limit=per_q))

                elif isinstance(searcher, AdzunaSearcher):
                    # Adzuna works best with short 1-3 word queries.
                    # The auto-generated query can be long compound strings like
                    # "chef cook kitchen food preparation" which return 0 results.
                    # Prefer the desired job title; fall back to the first 2 words
                    # of the primary query.
                    if profile.desired_titles:
                        adzuna_qs = [profile.desired_titles[0]]
                    else:
                        adzuna_qs = [" ".join(primary_query.split()[:2])]
                    per_q = max(20, 120 // max(1, len(adzuna_qs)))
                    for q in adzuna_qs:
                        _add(searcher.search(q, location=location, limit=per_q))

            except Exception as exc:
                logger.error("Error in source '%s': %s", source_name, exc)

        logger.info(
            "Total unique jobs: %d from %d source(s).", len(all_jobs), len(self._sources)
        )
        return all_jobs[:max_results]
