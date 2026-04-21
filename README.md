# 🔍 AutoJob Finder

An AI-powered automated job search application. Provide your **CV** (PDF/DOCX/TXT) or a **freeform description** of the job you want, and the app will:

1. Extract your skills, experience, and preferences automatically.
2. Search multiple job boards in parallel (LinkedIn/Indeed via JSearch, Adzuna, Remotive).
3. Rank every result by semantic match score and return your **top-k** jobs.
4. Let you open each application link directly from the UI.

---

## Features

| Feature | Detail |
|---|---|
| **CV parsing** | PDF, DOCX, TXT — structured extraction via GPT or regex fallback |
| **Multi-source search** | Remotive (free), JSearch/RapidAPI (LinkedIn + Indeed + Glassdoor …), Adzuna |
| **Smart ranking** | OpenAI embeddings → sentence-transformers → TF-IDF cosine (best available) |
| **Skill gap analysis** | Shows matched skills and skills to develop 📚 per job |
| **Filters** | Min score, remote-only, source filter |
| **One-click apply** | Direct link to the application page for every job |
| **No-key mode** | Works out of the box using the free Remotive API |

---

## Quick Start

### 1 — Clone / open the project

```bash
cd /home/dice/Automated_job_search
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~80 MB)
> on first use. If you supply an OpenAI key it is used instead and the model is
> never downloaded.

### 4 — Configure API keys (optional but recommended)

```bash
cp .env.example .env
# edit .env with your favourite editor
```

| Variable | Purpose | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Smart CV parsing + semantic matching | [platform.openai.com](https://platform.openai.com/api-keys) |
| `RAPIDAPI_KEY` | JSearch — LinkedIn, Indeed, Glassdoor, ZipRecruiter, … | [rapidapi.com → JSearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | Broad job coverage, free tier 250 req/day | [developer.adzuna.com](https://developer.adzuna.com/) |

> Keys can also be entered at runtime through the **sidebar** inside the app.

### 5 — Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Project Structure

```
Automated_job_search/
├── app.py                  ← Streamlit UI (entry point)
├── config.py               ← Settings (pydantic-settings, reads .env)
├── requirements.txt
├── .env.example            ← Template — copy to .env and fill in keys
│
└── src/
    ├── __init__.py
    ├── models.py           ← Pydantic data models (UserProfile, Job, RankedJob)
    ├── cv_parser.py        ← PDF/DOCX/TXT extraction + LLM / regex parsing
    ├── job_searcher.py     ← Remotive, JSearch (RapidAPI), Adzuna searchers
    └── job_matcher.py      ← Semantic + keyword ranking engine
```

---

## 🔧 How the Matching Works

```
Profile text  ──► Embedder ──► vector_p
                                          ──► cosine similarity ──► semantic score (0-100)
Job text      ──► Embedder ──► vector_j

Final score = 0.65 × semantic + 0.35 × keyword_overlap
            + skill_match_bonus  (up to +12 pts)
            + desired_title_bonus (+6 pts)
            + remote_bonus        (+3 pts)
            + location_bonus      (+3 pts)
```

**Embedder priority** (best available):
1. OpenAI `text-embedding-3-small` — requires `OPENAI_API_KEY`
2. `sentence-transformers/all-MiniLM-L6-v2` — local, ~80 MB
3. TF-IDF cosine — zero extra dependencies, always available

---

## 📸 App Flow

```
┌──────────────────────────────────────────────┐
│  Tab 1: Upload CV (PDF / DOCX / TXT)         │
│  Tab 2: Describe the job you want            │
│                                              │
│  Sidebar: API keys · Location · Remote · k  │
│                                              │
│  [ 🚀 Find My Top k Jobs ]                  │
└──────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────┐
│  Extracted Profile (expandable)              │
│  Fetching jobs …                             │
│  Ranking …                                   │
└──────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────┐
│  Results: top-k job cards                    │
│  Each card: title · company · location ·     │
│    salary · match % · matched skills ·       │
│    skills to develop · description ·         │
│    [ 🚀 Apply Now ]                          │
│  Filters: min score · remote · source        │
└──────────────────────────────────────────────┘
```

---

## 💡 Tips

* **No API keys?** The app works immediately using the free Remotive API (remote tech jobs).
* **Best results:** Add both `OPENAI_API_KEY` and `RAPIDAPI_KEY` — you'll get richer job data and more accurate ranking.
* **Tune k:** Use the sidebar slider to control how many top matches are shown (1–50).
* **Filters:** After results appear, use the "Filter results" expander to narrow by score, remote status, or source.
