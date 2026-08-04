"""
The Marks Man — backend API
----------------------------
A small FastAPI server that serves ONE endpoint the frontend needs:

    GET /api/news?limit=100&category=exams

It pulls fresh student-relevant stories from Google News RSS (free, no API
key needed), tags each story with one of the 6 categories the frontend
already knows about, keeps only stories from the last 7 days, and caches
the result in memory for a few minutes so we don't hammer Google News on
every page load.

Run it with:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
"""

import hashlib
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus, urlparse

import feedparser
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# 1. Config: one search query per category the frontend renders.
#    "when:7d" tells Google News to only return stories from the last week.
# ---------------------------------------------------------------------------

CATEGORY_QUERIES = {
    "exams": "exam datesheet OR hall ticket OR results India students",
    "campus_events": "college fest OR hackathon OR cultural fest India campus",
    "scholarships": "scholarship India students apply",
    "education_news": "education policy India university",
    "student_news": "student news India college",
    "kiit_news": "KIIT University Bhubaneswar",
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"

CACHE_TTL_SECONDS = 10 * 60  # refetch each category at most every 10 minutes
REQUEST_TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarksManBot/1.0)"}

# ---------------------------------------------------------------------------
# 2. In-memory cache: { category: {"articles": [...], "fetched_at": ts} }
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {}


def _strip_html(text: str) -> str:
    """Google News descriptions come wrapped in HTML/anchor tags — clean them up."""
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _domain_source(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "Unknown"


def _parse_published(entry) -> datetime:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def fetch_category(category: str, query: str) -> list[dict]:
    """Fetch + normalize one category's feed into our article schema."""
    url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    articles = []

    for entry in feed.entries:
        published = _parse_published(entry)
        if published < cutoff:
            continue

        link = entry.get("link", "")
        title = _strip_html(entry.get("title", "")).strip()
        if not link or not title:
            continue

        summary = _strip_html(entry.get("summary", ""))[:280]

        source_title = None
        if getattr(entry, "source", None) and entry.source.get("title"):
            source_title = entry.source["title"]

        articles.append({
            "id": _make_id(link),
            "title": title,
            "summary": summary,
            "url": link,
            "source": source_title or _domain_source(link),
            "category": category,
            "image_url": None,  # Google News RSS doesn't expose thumbnails
            "published_at": published.isoformat(),
        })

    return articles


def get_category_articles(category: str, query: str) -> list[dict]:
    """Cache-aware fetch: only re-hits Google News if the cache is stale."""
    cached = _cache.get(category)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["articles"]

    fresh = fetch_category(category, query)
    # Fall back to the old cached copy if the fetch failed / returned nothing,
    # so a transient network hiccup doesn't blank out a whole section.
    if not fresh and cached:
        fresh = cached["articles"]

    _cache[category] = {"articles": fresh, "fetched_at": time.time()}
    return fresh


def get_all_articles() -> list[dict]:
    all_articles = []
    for category, query in CATEGORY_QUERIES.items():
        all_articles.extend(get_category_articles(category, query))
    all_articles.sort(key=lambda a: a["published_at"], reverse=True)
    return all_articles


# ---------------------------------------------------------------------------
# 3. FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="The Marks Man API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # keep permissive since this feeds a static HTML page
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/news")
def get_news(
    limit: int = Query(100, ge=1, le=200),
    category: str | None = Query(None, description="Filter to one category key"),
):
    if category:
        articles = get_category_articles(category, CATEGORY_QUERIES.get(category, category))
    else:
        articles = get_all_articles()

    return articles[:limit]


@app.get("/api/health")
def health():
    return {"status": "ok", "categories": list(CATEGORY_QUERIES.keys())}