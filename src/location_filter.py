"""
Location-aware scoring utility.

Given a user's desired location and a job's location string, returns a
numeric score delta and an optional human-readable reason.

Score deltas
────────────────────────────────────────────────────
 +8   exact city / region match
 +4   same country
 +2   same continent
  0   remote worldwide / unknown location
-12   remote but explicitly restricted to wrong region
-20   onsite in a different continent
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Data tables
# ─────────────────────────────────────────────────────────────────

# City → canonical country name
_CITY_TO_COUNTRY: dict[str, str] = {
    # Germany
    "munich": "germany", "münchen": "germany", "munchen": "germany",
    "berlin": "germany", "hamburg": "germany", "cologne": "germany",
    "köln": "germany", "koln": "germany", "frankfurt": "germany",
    "stuttgart": "germany", "düsseldorf": "germany", "dusseldorf": "germany",
    "dortmund": "germany", "essen": "germany", "bremen": "germany",
    "hannover": "germany", "nuremberg": "germany", "nürnberg": "germany",
    "leipzig": "germany", "dresden": "germany",
    # Austria
    "vienna": "austria", "wien": "austria", "graz": "austria", "salzburg": "austria",
    # Switzerland
    "zurich": "switzerland", "zürich": "switzerland", "zurich": "switzerland",
    "geneva": "switzerland", "genf": "switzerland", "bern": "switzerland",
    "basel": "switzerland",
    # UK
    "london": "uk", "manchester": "uk", "edinburgh": "uk", "birmingham": "uk",
    "bristol": "uk", "leeds": "uk", "glasgow": "uk", "liverpool": "uk",
    # France
    "paris": "france", "lyon": "france", "marseille": "france",
    "toulouse": "france", "bordeaux": "france", "lille": "france",
    # Netherlands
    "amsterdam": "netherlands", "rotterdam": "netherlands", "the hague": "netherlands",
    "den haag": "netherlands", "utrecht": "netherlands", "eindhoven": "netherlands",
    # Spain
    "barcelona": "spain", "madrid": "spain", "valencia": "spain", "seville": "spain",
    # Italy
    "rome": "italy", "milan": "italy", "milano": "italy", "turin": "italy",
    "torino": "italy", "naples": "italy", "florence": "italy",
    # Belgium
    "brussels": "belgium", "bruxelles": "belgium", "antwerp": "belgium",
    # Poland
    "warsaw": "poland", "krakow": "poland", "wroclaw": "poland",
    # Czech Republic
    "prague": "czech republic", "brno": "czech republic",
    # Hungary
    "budapest": "hungary",
    # Portugal
    "lisbon": "portugal", "lisboa": "portugal", "porto": "portugal",
    # Sweden
    "stockholm": "sweden", "gothenburg": "sweden", "göteborg": "sweden",
    # Norway
    "oslo": "norway", "bergen": "norway",
    # Denmark
    "copenhagen": "denmark", "københavn": "denmark",
    # Finland
    "helsinki": "finland", "tampere": "finland",
    # Greece
    "athens": "greece", "thessaloniki": "greece",
    # Romania
    "bucharest": "romania", "cluj": "romania",
    # USA
    "new york": "usa", "nyc": "usa", "new york city": "usa",
    "san francisco": "usa", "sf": "usa", "bay area": "usa",
    "silicon valley": "usa", "los angeles": "usa", "la": "usa",
    "seattle": "usa", "boston": "usa", "chicago": "usa", "austin": "usa",
    "denver": "usa", "miami": "usa", "atlanta": "usa",
    "washington": "usa", "dc": "usa", "washington dc": "usa",
    "new jersey": "usa", "nj": "usa", "palo alto": "usa",
    "mountain view": "usa", "menlo park": "usa",
    # Canada
    "toronto": "canada", "vancouver": "canada", "montreal": "canada",
    "calgary": "canada", "ottawa": "canada",
    # Australia
    "sydney": "australia", "melbourne": "australia", "brisbane": "australia",
    "perth": "australia", "adelaide": "australia",
    # Asia
    "tokyo": "japan", "osaka": "japan", "kyoto": "japan",
    "singapore": "singapore",
    "beijing": "china", "shanghai": "china", "shenzhen": "china",
    "hong kong": "hong kong",
    "seoul": "south korea", "busan": "south korea",
    "mumbai": "india", "bangalore": "india", "bengaluru": "india",
    "delhi": "india", "new delhi": "india", "hyderabad": "india",
    "pune": "india", "chennai": "india",
    # Middle East
    "dubai": "uae", "abu dhabi": "uae",
    "tel aviv": "israel",
    # Africa
    "cape town": "south africa", "johannesburg": "south africa",
    "nairobi": "kenya", "lagos": "nigeria",
}

# Country → continent
_COUNTRY_TO_CONTINENT: dict[str, str] = {
    # Europe
    "germany": "europe", "deutschland": "europe", "austria": "europe",
    "switzerland": "europe", "uk": "europe", "united kingdom": "europe",
    "france": "europe", "netherlands": "europe", "spain": "europe",
    "italy": "europe", "belgium": "europe", "poland": "europe",
    "czech republic": "europe", "czechia": "europe", "hungary": "europe",
    "portugal": "europe", "sweden": "europe", "norway": "europe",
    "denmark": "europe", "finland": "europe", "greece": "europe",
    "romania": "europe", "bulgaria": "europe", "croatia": "europe",
    "slovakia": "europe", "slovenia": "europe", "ireland": "europe",
    "latvia": "europe", "lithuania": "europe", "estonia": "europe",
    "luxembourg": "europe", "malta": "europe", "cyprus": "europe",
    "serbia": "europe", "ukraine": "europe", "russia": "europe",
    "eu": "europe", "europe": "europe", "europa": "europe",
    "emea": "europe",  # loose but useful
    # Americas
    "usa": "americas", "united states": "americas", "us": "americas",
    "canada": "americas", "mexico": "americas", "brazil": "americas",
    "argentina": "americas", "colombia": "americas", "chile": "americas",
    "north america": "americas", "latin america": "americas",
    # Asia-Pacific
    "japan": "asia", "china": "asia", "south korea": "asia", "korea": "asia",
    "india": "asia", "singapore": "asia", "hong kong": "asia",
    "taiwan": "asia", "thailand": "asia", "indonesia": "asia",
    "malaysia": "asia", "vietnam": "asia", "philippines": "asia",
    "pakistan": "asia", "bangladesh": "asia",
    "australia": "oceania", "new zealand": "oceania",
    # Middle East
    "uae": "middle east", "israel": "middle east", "saudi arabia": "middle east",
    "turkey": "middle east",
    # Africa
    "south africa": "africa", "kenya": "africa", "nigeria": "africa",
    "egypt": "africa", "ethiopia": "africa",
}

# Terms that explicitly mean "this remote job restricts to a region"
_REGION_RESTRICTION_PATTERNS = [
    (re.compile(r"\b(us\s*only|usa\s*only|north\s*america\s*only|americas\s*only"
                r"|united\s*states\s*only)\b", re.IGNORECASE), "americas"),
    (re.compile(r"\b(europe\s*only|eu\s*only|emea\s*only|european\s*union\s*only)\b",
                re.IGNORECASE), "europe"),
    (re.compile(r"\b(asia\s*only|apac\s*only|asia.pacific\s*only)\b",
                re.IGNORECASE), "asia"),
    (re.compile(r"\b(uk\s*only|united\s*kingdom\s*only)\b", re.IGNORECASE), "uk"),
    (re.compile(r"\b(canada\s*only|canadian\s*only)\b", re.IGNORECASE), "canada"),
    (re.compile(r"\b(australia\s*only|au\s*only)\b", re.IGNORECASE), "oceania"),
    # Positive region hints (remote but based in a region)
    (re.compile(r"\b(remote\s*(?:in\s*)?europe|europe[ans]*\s*remote)\b",
                re.IGNORECASE), "europe"),
    (re.compile(r"\b(remote\s*(?:in\s*)?(?:usa|us|north\s*america))\b",
                re.IGNORECASE), "americas"),
    (re.compile(r"\b(remote\s*(?:in\s*)?(?:uk|united\s*kingdom))\b",
                re.IGNORECASE), "uk"),
    (re.compile(r"\b(remote\s*(?:in\s*)?(?:germany|deutschland))\b",
                re.IGNORECASE), "germany"),
]

# Neutral / worldwide terms
_WORLDWIDE_RE = re.compile(
    r"\b(worldwide|anywhere|global|international|anywhere\s+in\s+the\s+world"
    r"|all\s+countries|location\s+independent)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


# ─────────────────────────────────────────────────────────────────
# Region resolution
# ─────────────────────────────────────────────────────────────────

def _resolve(location_text: str) -> dict:
    """
    Return a dict with keys: city, country, continent (all lower-case str or None).
    """
    norm = _normalize(location_text)
    result = {"city": None, "country": None, "continent": None}

    # Try city lookup (longest match first)
    for city in sorted(_CITY_TO_COUNTRY, key=len, reverse=True):
        if re.search(r"\b" + re.escape(city) + r"\b", norm):
            result["city"] = city
            result["country"] = _CITY_TO_COUNTRY[city]
            break

    # Try direct country match if no city found
    if result["country"] is None:
        for country in sorted(_COUNTRY_TO_CONTINENT, key=len, reverse=True):
            if re.search(r"\b" + re.escape(country) + r"\b", norm):
                result["country"] = country
                break

    if result["country"]:
        result["continent"] = _COUNTRY_TO_CONTINENT.get(result["country"])

    return result


# ─────────────────────────────────────────────────────────────────
# Public scoring function
# ─────────────────────────────────────────────────────────────────

def score_location_match(
    desired: str,
    job_location: Optional[str],
    job_remote: bool,
    job_description: str = "",
) -> Tuple[float, str]:
    """
    Return ``(score_delta, reason)`` where:
    - ``score_delta`` is added to the job's base match score (can be negative)
    - ``reason`` is a human-readable label (empty string if neutral)

    Parameters
    ----------
    desired:         The user's preferred location string (e.g. "Munich").
    job_location:    The job's extracted location field (may be None / "See description").
    job_remote:      Whether the job is flagged as remote.
    job_description: Raw job description text (used to detect regional restrictions).
    """
    if not desired or not desired.strip():
        return 0.0, ""

    desired_info = _resolve(desired)
    desired_city      = desired_info["city"]
    desired_country   = desired_info["country"]
    desired_continent = desired_info["continent"]

    # ── Determine the job's own location info ────────────────────
    loc_text = " ".join(filter(None, [job_location, job_description[:400]]))
    loc_norm  = _normalize(loc_text)

    # Detect worldwide / neutral remote
    if job_remote and _WORLDWIDE_RE.search(loc_norm):
        return 0.0, ""   # neutral – let semantic score decide

    # Detect explicit regional restrictions in remote text
    if job_remote:
        for pattern, restricted_to in _REGION_RESTRICTION_PATTERNS:
            if pattern.search(loc_norm):
                # Does the restriction match the user's continent / country?
                if (
                    desired_continent and restricted_to == desired_continent
                    or desired_country and restricted_to == desired_country
                ):
                    return +3.0, f"Remote ({restricted_to.title()} friendly)"
                else:
                    return -12.0, f"Remote restricted to {restricted_to.title()}"

    job_info = _resolve(loc_text)
    job_city      = job_info["city"]
    job_country   = job_info["country"]
    job_continent = job_info["continent"]

    # ── No location info at all ──────────────────────────────────
    if job_city is None and job_country is None and job_continent is None:
        if job_remote:
            return 0.0, ""   # unknown but remote = neutral
        return 0.0, ""       # truly unknown = neutral (don't penalise)

    # ── City match ───────────────────────────────────────────────
    if desired_city and job_city and desired_city == job_city:
        return +8.0, f"📍 In {desired.title()}"

    # ── Country match ────────────────────────────────────────────
    if desired_country and job_country and desired_country == job_country:
        return +4.0, f"📍 In {job_country.title()}"

    # ── Continent match ──────────────────────────────────────────
    if desired_continent and job_continent and desired_continent == job_continent:
        return +2.0, f"📍 In {job_continent.title()}"

    # ── Different continent ──────────────────────────────────────
    if desired_continent and job_continent and desired_continent != job_continent:
        if job_remote:
            return -8.0, f"Remote (based {job_continent.title()}, you want {desired_continent.title()})"
        return -20.0, f"⚠️ Onsite in {job_continent.title()} (you want {desired_continent.title()})"

    # ── Different country, same continent ────────────────────────
    if desired_country and job_country and desired_country != job_country:
        if job_remote:
            return -4.0, ""
        return -10.0, ""

    return 0.0, ""
