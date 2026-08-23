"""app/ingestion.py

Consolidated ingestion pipeline: URL dedup hashing, RSS fetching,
NewsAPI fetching, and the run_ingestion orchestration that upserts
new articles and prunes old ones.

This replaces the former app/ingestion/ package (dedup.py,
rss_fetcher.py, api_fetcher.py, pipeline.py, __init__.py) with a
single module. If anything else in the codebase does
`from app.ingestion.rss_fetcher import ...` etc., update those
imports to `from app.ingestion import ...`.
"""

from datetime import datetime, timedelta, timezone
import hashlib

import feedparser
import httpx
from sqlalchemy.orm import Session

from app.config import CATEGORIES, settings
from app.database import SessionLocal
from app.models import Article

NEWSAPI_URL = "https://newsapi.org/v2/everything"


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------

def url_hash(url: str) -> str:
    """Stable hash used as the dedup key so re-ingesting the same
    article (e.g. on the next scheduled run) doesn't create duplicates.
    """
    normalized = url.strip().lower().rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# RSS fetching
# --------------------------------------------------------------------------

def fetch_rss_for_category(category_key: str) -> list[dict]:
    """Pull articles from every RSS feed configured for a category.
    Returns a list of dicts ready to be upserted as Article rows.
    """
    feeds = CATEGORIES.get(category_key, {}).get("rss_feeds", [])
    results: list[dict] = []

    for feed_url in feeds:
        parsed = feedparser.parse(feed_url)
        source_name = parsed.feed.get("title", feed_url)

        for entry in parsed.entries:
            link = entry.get("link")
            if not link:
                continue

            published = entry.get("published_parsed") or entry.get("updated_parsed")
            published_at = (
                datetime(*published[:6], tzinfo=timezone.utc)
                if published
                else datetime.now(timezone.utc)
            )

            results.append(
                {
                    "title": entry.get("title", "Untitled"),
                    "url": link,
                    "url_hash": url_hash(link),
                    "source": source_name,
                    "category": category_key,
                    "summary": entry.get("summary", "")[:1000],
                    "image_url": _extract_rss_image(entry),
                    "published_at": published_at,
                }
            )

    return results


def _extract_rss_image(entry) -> str | None:
    media = entry.get("media_content") or entry.get("media_thumbnail")
    if media and isinstance(media, list) and "url" in media[0]:
        return media[0]["url"]
    return None


# --------------------------------------------------------------------------
# NewsAPI fetching
# --------------------------------------------------------------------------

def fetch_newsapi_for_category(category_key: str) -> list[dict]:
    """Query NewsAPI using the category's keyword list. Requires
    NEWSAPI_KEY to be set; returns [] silently if it isn't, so the
    ingestion pipeline degrades gracefully to RSS-only.
    """
    if not settings.newsapi_key:
        return []

    keywords = CATEGORIES.get(category_key, {}).get("keywords", [])
    if not keywords:
        return []

    query = " OR ".join(keywords)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 30,
        "apiKey": settings.newsapi_key,
    }

    try:
        response = httpx.get(NEWSAPI_URL, params=params, timeout=15)
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    articles = response.json().get("articles", [])
    results = []

    for a in articles:
        link = a.get("url")
        if not link:
            continue
        published_raw = a.get("publishedAt")
        try:
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            published_at = datetime.now(timezone.utc)

        results.append(
            {
                "title": a.get("title", "Untitled"),
                "url": link,
                "url_hash": url_hash(link),
                "source": (a.get("source") or {}).get("name", "Unknown"),
                "category": category_key,
                "summary": (a.get("description") or "")[:1000],
                "image_url": a.get("urlToImage"),
                "published_at": published_at,
            }
        )

    return results


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_ingestion() -> dict:
    """Fetches new articles for every category, inserts anything not
    already seen (by url_hash), then prunes anything past the
    retention window. Call this from the scheduler or the
    /internal/refresh endpoint.
    """
    db: Session = SessionLocal()
    inserted = 0
    try:
        for category_key in CATEGORIES:
            items = fetch_rss_for_category(category_key) + fetch_newsapi_for_category(category_key)
            for item in items:
                inserted += _upsert_article(db, item)
        db.commit()

        pruned = _prune_old_articles(db)
        db.commit()
    finally:
        db.close()

    return {"inserted": inserted, "pruned": pruned}


def _upsert_article(db: Session, item: dict) -> int:
    exists = db.query(Article.id).filter(Article.url_hash == item["url_hash"]).first()
    if exists:
        return 0
    db.add(Article(**item))
    return 1


def _prune_old_articles(db: Session) -> int:
    cutoff = datetime.utcnow() - timedelta(days=settings.news_retention_days)
    result = db.query(Article).filter(Article.published_at < cutoff).delete()
    return result
