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
import html
import re

import feedparser
import httpx
from sqlalchemy.orm import Session

from app import CATEGORIES, settings
from app import SessionLocal
from app import Article

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

def clean_html(text: str) -> str:
    """Strips HTML tags, unescapes entities, and collapses whitespace."""
    if not text:
        return ""
    unescaped = html.unescape(str(text))
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


def extract_rss_image(entry) -> str | None:
    """Robustly extracts image URL from media_content, media_thumbnail,
    enclosures, links, or inline HTML img tags.
    """
    if entry.get("media_content") and isinstance(entry["media_content"], list) and "url" in entry["media_content"][0]:
        return entry["media_content"][0]["url"]
    if entry.get("media_thumbnail") and isinstance(entry["media_thumbnail"], list) and "url" in entry["media_thumbnail"][0]:
        return entry["media_thumbnail"][0]["url"]
    for enc in entry.get("enclosures", []):
        href = enc.get("href", "")
        if href and ("image" in enc.get("type", "") or any(href.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"])):
            return href
    for l in entry.get("links", []):
        href = l.get("href", "")
        if href and ("image" in l.get("type", "") or l.get("rel") == "enclosure"):
            return href
    text = (entry.get("summary") or "") + " " + (entry.get("description") or "")
    m = re.search(r'src=["\'](https?://[^"\'\s]+\.(?:jpg|jpeg|png|webp)[^"\'\s]*)["\']', text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def fetch_rss_for_category(category_key: str) -> list[dict]:
    """Pull articles from every RSS feed configured for a category.
    Returns a list of dicts ready to be upserted as Article rows.
    """
    feeds = CATEGORIES.get(category_key, {}).get("rss_feeds", [])
    results: list[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    for feed_url in feeds:
        try:
            resp = httpx.get(feed_url, timeout=10, headers=headers, follow_redirects=True)
            parsed = feedparser.parse(resp.content)
        except Exception:
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

            image_url = extract_rss_image(entry)

            raw_summary = entry.get("summary") or entry.get("description") or ""
            cleaned_summary = clean_html(raw_summary)[:1000]

            results.append(
                {
                    "title": clean_html(entry.get("title", "Untitled")),
                    "url": link,
                    "url_hash": url_hash(link),
                    "source": source_name,
                    "category": category_key,
                    "summary": cleaned_summary,
                    "image_url": image_url,
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


GNEWS_URL = "https://gnews.io/api/v4/search"


def fetch_gnews_for_category(category_key: str) -> list[dict]:
    """Query GNews API using category keywords if GNEWS_API_KEY is configured."""
    if not settings.gnews_api_key:
        return []

    keywords = CATEGORIES.get(category_key, {}).get("keywords", [])
    if not keywords:
        return []

    query = " OR ".join(keywords[:5])
    params = {
        "q": query,
        "lang": "en",
        "country": "in",
        "max": 20,
        "apikey": settings.gnews_api_key,
    }

    try:
        response = httpx.get(GNEWS_URL, params=params, timeout=15)
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
                "title": clean_html(a.get("title", "Untitled")),
                "url": link,
                "url_hash": url_hash(link),
                "source": (a.get("source") or {}).get("name", "GNews"),
                "category": category_key,
                "summary": clean_html(a.get("description") or "")[:1000],
                "image_url": a.get("image"),
                "published_at": published_at,
            }
        )

    return results


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_ingestion() -> dict:
    """Fetches new articles for every category from Google News RSS feeds,
    NewsAPI, and GNews API, inserts anything not already seen (by url_hash),
    then prunes anything past the retention window. Call this from the scheduler or the
    /internal/refresh endpoint.
    """
    db: Session = SessionLocal()
    inserted = 0
    pruned = 0
    try:
        seen_hashes = {h for (h,) in db.query(Article.url_hash).all()}
        for category_key in CATEGORIES:
            items = (
                fetch_rss_for_category(category_key)
                + fetch_newsapi_for_category(category_key)
                + fetch_gnews_for_category(category_key)
            )
            for item in items:
                u_hash = item.get("url_hash")
                if not u_hash or u_hash in seen_hashes:
                    continue
                seen_hashes.add(u_hash)
                try:
                    db.add(Article(**item))
                    db.commit()
                    inserted += 1
                except Exception:
                    db.rollback()

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=settings.news_retention_days)
            pruned = db.query(Article).filter(Article.published_at < cutoff).delete()
            db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()

    return {"inserted": inserted, "pruned": pruned}
