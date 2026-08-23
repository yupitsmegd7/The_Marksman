"""app.py

Fully standalone application module: settings/categories, DB engine/
session, the Article model, Pydantic schemas, ingestion (RSS + API
fetchers, dedup, the retention-pruning pipeline), NLP (spaCy-based
extractive summarization, YAKE keyword extraction, displacy
visualization), the background scheduler, and the FastAPI app itself
— all in this one file. No other app/* modules are needed; nothing
here imports from `app.ingestion` or `app.nlp`.

Run with: uvicorn app:app --reload

Install:
  pip install fastapi uvicorn[standard] sqlalchemy psycopg2-binary \
      pydantic pydantic-settings apscheduler feedparser httpx \
      beautifulsoup4 python-dotenv trafilatura yake spacy numpy networkx
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
  # (or: python -m spacy download en_core_web_md, if the direct wheel
  # URL is blocked by your pip/network config)
"""

import hashlib
import re
from datetime import datetime, timedelta, timezone

import httpx
import networkx as nx
import numpy as np
import spacy
import trafilatura
import yake
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from spacy import displacy
import feedparser
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy import Column, DateTime, Index, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker


# --------------------------------------------------------------------------
# Settings & categories
# --------------------------------------------------------------------------

class Settings(BaseSettings):
    database_url: str = "sqlite:///./news_portal.db"
    frontend_origin: str = "http://localhost:5173"
    newsapi_key: str = ""
    gnews_api_key: str = ""
    news_retention_days: int = 7
    ingestion_interval_minutes: int = 120
    internal_refresh_token: str = "change-me"

    class Config:
        env_file = ".env"


settings = Settings()

# Categories drive both ingestion sources and frontend sub-sections.
# Add/remove entries here to add a new "sub-site" without touching the schema.
CATEGORIES = {
    "exams": {
        "label": "Exams",
        "keywords": ["exam", "results", "admit card", "board exam", "university exam"],
        "rss_feeds": [
            # Add real RSS feeds for exam boards / universities you track
        ],
    },
    "campus_events": {
        "label": "Campus Events",
        "keywords": ["campus", "fest", "hackathon", "convocation", "workshop"],
        "rss_feeds": [],
    },
    "scholarships": {
        "label": "Scholarships",
        "keywords": ["scholarship", "internship", "fellowship", "stipend"],
        "rss_feeds": [],
    },
    "education_news": {
        "label": "Education News",
        "keywords": ["education policy", "education technology", "online learning", "university"],
        "rss_feeds": [],
    },
    "student_news": {
        "label": "Student News",
        "keywords": ["student", "student union", "youth", "college student"],
        "rss_feeds": [],
    },
    "kiit_news": {
        "label": "KIIT News",
        "keywords": ["KIIT", "KIIT University", "KIIT Bhubaneswar"],
        "rss_feeds": [
            # Add KIIT's official announcements/press RSS feed here if available
        ],
    },
}


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False)
    url_hash = Column(String(64), nullable=False)  # dedup key, see url_hash() below
    source = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False, index=True)
    summary = Column(Text, nullable=True)
    image_url = Column(String(1000), nullable=True)
    published_at = Column(DateTime, nullable=False, index=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Populated on-demand (lazily, cached) the first time a reader opens
    # the headline popup — see get_or_build_insights() and the
    # /insights endpoint below.
    full_text = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True)
    keywords = Column(String(500), nullable=True)  # comma-separated

    __table_args__ = (
        UniqueConstraint("url_hash", name="uq_article_url_hash"),
        Index("ix_category_published", "category", "published_at"),
    )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ArticleOut(BaseModel):
    id: int
    title: str
    url: str
    source: str
    category: str
    summary: str | None = None
    image_url: str | None = None
    published_at: datetime

    class Config:
        from_attributes = True


class CategoryOut(BaseModel):
    key: str
    label: str


class InsightsOut(BaseModel):
    summary: str
    keywords: list[str]
    source: str  # "cached" | "generated" | "fallback"


# --------------------------------------------------------------------------
# NLP: spaCy model (loaded once at import time, not per-request)
# --------------------------------------------------------------------------

# "md" (medium) is the smallest spaCy English model that ships real word
# vectors — "sm" doesn't have them, and .similarity() silently degrades
# to a much weaker signal on it. This is a ~40MB model held in memory
# for the life of the process; a real but modest cost versus the
# TF-IDF version of this pipeline, which needed nothing pre-loaded.
nlp = spacy.load("en_core_web_md")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def build_similarity_matrix(docs) -> np.ndarray:
    """Pairwise semantic similarity between sentences, using spaCy's
    pretrained word vectors instead of raw word overlap (TF-IDF).
    Two sentences that say the same thing in different words (e.g.
    "university" vs. "college") now score as similar, where TF-IDF
    would have seen zero shared words and scored them as unrelated.
    """
    n = len(docs)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # A sentence with no recognizable words (rare, but possible)
            # has a zero vector — comparing against it isn't meaningful
            # and triggers a spaCy warning, so skip it.
            if docs[i].vector_norm and docs[j].vector_norm:
                matrix[i][j] = docs[i].similarity(docs[j])
    return matrix


def summarize(text: str, num_sentences: int = 3) -> str:
    """TextRank-style extractive summary: builds a sentence similarity
    graph using spaCy semantic vectors, ranks sentences by PageRank
    (the same core algorithm Google originally used to rank web
    pages, applied here to sentences instead of pages), and returns
    the top N sentences in their ORIGINAL reading order.

    Fully extractive: every returned sentence is copied verbatim from
    the source article. Nothing is generated or reworded.
    """
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    docs = [nlp(sentence) for sentence in sentences]
    similarity_matrix = build_similarity_matrix(docs)

    graph = nx.from_numpy_array(similarity_matrix)
    try:
        scores = nx.pagerank(graph)
    except nx.PowerIterationFailedConvergence:
        return " ".join(sentences[:num_sentences])

    ranked_idx = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    ranked_idx.sort()  # restore original reading order
    return " ".join(sentences[i] for i in ranked_idx)


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """YAKE is unsupervised and statistical (no pretrained model to
    download), which keeps this fast and deploy-friendly. Returns
    keywords/short phrases ranked by relevance, best first.
    """
    if not text or not text.strip():
        return []

    extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=max_keywords)
    ranked = extractor.extract_keywords(text)
    # yake returns (keyword, score) with LOWER score = more relevant
    ranked.sort(key=lambda pair: pair[1])
    return [kw for kw, _score in ranked][:max_keywords]


def fetch_full_text(url: str, timeout: int = 12) -> str | None:
    """Downloads the article page and pulls out just the main body
    text (strips nav, ads, comments, etc). Returns None if the fetch
    or extraction fails — callers should fall back to the RSS/API
    summary in that case rather than erroring out the popup.
    """
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TheMarksManBot/1.0)"},
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
    return text.strip() if text else None


def get_or_build_insights(db: Session, article: Article) -> dict:
    """Returns {summary, keywords, source: 'cached'|'generated'|'fallback'}.

    Cached on the Article row after first generation, so re-opening
    the same headline's popup is instant and doesn't re-hit the NLP
    pipeline or re-scrape the source page.
    """
    if article.ai_summary and article.keywords:
        return {
            "summary": article.ai_summary,
            "keywords": article.keywords.split(","),
            "source": "cached",
        }

    full_text = fetch_full_text(article.url)
    text_for_nlp = full_text or article.summary or article.title

    if not text_for_nlp:
        return {"summary": "", "keywords": [], "source": "fallback"}

    generated_summary = summarize(text_for_nlp, num_sentences=3)
    generated_keywords = extract_keywords(text_for_nlp, max_keywords=8)

    article.full_text = full_text
    article.ai_summary = generated_summary
    article.keywords = ",".join(generated_keywords)
    db.add(article)
    db.commit()

    return {
        "summary": generated_summary,
        "keywords": generated_keywords,
        "source": "generated" if full_text else "fallback",
    }


def render_entities_html(text: str) -> str:
    """Highlights named entities (people, orgs, dates, locations...)
    in colored spans — good for spotting at a glance who/what/when a
    story is about.
    """
    doc = nlp(text)
    return displacy.render(doc, style="ent", page=True)


def render_dependency_html(text: str) -> str:
    """Draws the grammatical dependency tree — arrows showing how
    words relate to each other in a sentence. Most readable on a
    single sentence at a time; rendering several sentences produces
    one diagram per sentence, stacked.
    """
    doc = nlp(text)
    return displacy.render(doc, style="dep", page=True, options={"compact": True, "distance": 110})


# --------------------------------------------------------------------------
# Ingestion: dedup, RSS fetcher, API fetcher, retention pipeline
# --------------------------------------------------------------------------

def url_hash(url: str) -> str:
    """Stable hash used as the dedup key so re-ingesting the same
    article (e.g. on the next scheduled run) doesn't create duplicates.
    """
    normalized = url.strip().lower().rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fetch_rss_for_category(category_key: str) -> list[dict]:
    """Pull articles from every RSS feed configured for a category."""
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
                datetime(*published[:6], tzinfo=timezone.utc) if published else datetime.now(timezone.utc)
            )

            media = entry.get("media_content") or entry.get("media_thumbnail")
            image_url = media[0]["url"] if media and isinstance(media, list) and "url" in media[0] else None

            results.append(
                {
                    "title": entry.get("title", "Untitled"),
                    "url": link,
                    "url_hash": url_hash(link),
                    "source": source_name,
                    "category": category_key,
                    "summary": entry.get("summary", "")[:1000],
                    "image_url": image_url,
                    "published_at": published_at,
                }
            )

    return results


NEWSAPI_URL = "https://newsapi.org/v2/everything"


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


def run_ingestion() -> dict:
    """Fetches new articles for every category, inserts anything not
    already seen (by url_hash), then prunes anything past the
    retention window. Called by the scheduler and by /internal/refresh.
    """
    db: Session = SessionLocal()
    inserted = 0
    try:
        for category_key in CATEGORIES:
            items = fetch_rss_for_category(category_key) + fetch_newsapi_for_category(category_key)
            for item in items:
                exists = db.query(Article.id).filter(Article.url_hash == item["url_hash"]).first()
                if exists:
                    continue
                db.add(Article(**item))
                inserted += 1
        db.commit()

        cutoff = datetime.utcnow() - timedelta(days=settings.news_retention_days)
        pruned = db.query(Article).filter(Article.published_at < cutoff).delete()
        db.commit()
    finally:
        db.close()

    return {"inserted": inserted, "pruned": pruned}


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------

scheduler = BackgroundScheduler()


def start_scheduler():
    """In-process scheduler for local dev / always-on hosts.

    NOTE: on free/serverless hosting tiers that spin down when idle,
    this won't fire reliably — use the platform's own cron feature to
    hit POST /internal/refresh instead.
    """
    scheduler.add_job(
        run_ingestion,
        "interval",
        minutes=settings.ingestion_interval_minutes,
        id="news_ingestion",
        replace_existing=True,
    )
    scheduler.start()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(title="The Marks Man API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    start_scheduler()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/internal/refresh")
def internal_refresh(x_refresh_token: str = Header(default="")):
    """Hit this from an external cron job (Render/Railway Cron) if
    the host's free tier doesn't keep the in-process scheduler alive.
    """
    if x_refresh_token != settings.internal_refresh_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return run_ingestion()


# --------------------------------------------------------------------------
# News routes
# --------------------------------------------------------------------------

@app.get("/api/categories", response_model=list[CategoryOut], tags=["news"])
def list_categories():
    return [{"key": key, "label": val["label"]} for key, val in CATEGORIES.items()]


@app.get("/api/news", response_model=list[ArticleOut], tags=["news"])
def get_news(
    category: str | None = Query(default=None),
    limit: int = Query(default=30, le=100),
    db: Session = Depends(get_db),
):
    cutoff = datetime.utcnow() - timedelta(days=settings.news_retention_days)
    query = db.query(Article).filter(Article.published_at >= cutoff)

    if category:
        query = query.filter(Article.category == category)

    return query.order_by(Article.published_at.desc()).limit(limit).all()


@app.get("/api/news/latest", response_model=list[ArticleOut], tags=["news"])
def get_latest(limit: int = Query(default=10, le=50), db: Session = Depends(get_db)):
    cutoff = datetime.utcnow() - timedelta(days=settings.news_retention_days)
    return (
        db.query(Article)
        .filter(Article.published_at >= cutoff)
        .order_by(Article.published_at.desc())
        .limit(limit)
        .all()
    )


@app.get("/api/articles/{article_id}/insights", response_model=InsightsOut, tags=["news"])
def get_article_insights(article_id: int, db: Session = Depends(get_db)):
    """Powers the headline popup: fetches the full article, runs it
    through extractive summarization + keyword extraction, and caches
    the result on first request so later clicks are instant.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return get_or_build_insights(db, article)


@app.get("/api/articles/{article_id}/visualize", response_class=HTMLResponse, tags=["news"])
def visualize_article(
    article_id: int,
    style: str = Query(default="ent", pattern="^(ent|dep)$"),
    db: Session = Depends(get_db),
):
    """Renders the article's summary through spaCy's displacy —
    style=ent highlights named entities (people, orgs, dates...),
    style=dep draws the grammatical dependency tree. Returns a full
    standalone HTML page (displacy's own output), meant to be opened
    directly rather than embedded.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    # Visualize the (short) summary rather than the raw article — a
    # multi-sentence dependency diagram of a full article gets very
    # wide and hard to read.
    insights = get_or_build_insights(db, article)
    text = insights["summary"] or article.title

    html = render_dependency_html(text) if style == "dep" else render_entities_html(text)
    return HTMLResponse(content=html)
