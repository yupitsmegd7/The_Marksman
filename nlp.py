"""app/nlp.py

Consolidated NLP pipeline: full-text extraction, extractive
summarization (TextRank-style), keyword extraction, and the
DB-caching orchestration layer that ties them together.

This replaces the former app/nlp/ package (extractor.py, keywords.py,
summarizer.py, pipeline.py, __init__.py) with a single module. If
anything else in the codebase does `from app.nlp.extractor import ...`
etc., update those imports to `from app.nlp import ...`.
"""

import re

import httpx
import networkx as nx
import trafilatura
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from app.models import Article

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


# --------------------------------------------------------------------------
# Full-text extraction
# --------------------------------------------------------------------------

def fetch_full_text(url: str, timeout: int = 12) -> str|None:
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


# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def summarize(text: str, num_sentences: int = 3) -> str:
    """TextRank-style extractive summary: builds a sentence similarity
    graph with TF-IDF + cosine similarity, ranks sentences by
    PageRank, and returns the top N in their original order (so the
    summary still reads coherently rather than as a shuffled list).

    Deliberately avoids transformer models (e.g. BART/T5) so this can
    run comfortably on a free-tier web dyno without a GPU or a slow
    cold-start model download.
    """
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        # e.g. text is all stopwords / too short after cleaning
        return " ".join(sentences[:num_sentences])

    similarity_matrix = cosine_similarity(tfidf)
    graph = nx.from_numpy_array(similarity_matrix)

    try:
        scores = nx.pagerank(graph)
    except nx.PowerIterationFailedConvergence:
        return " ".join(sentences[:num_sentences])

    ranked_idx = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    ranked_idx.sort()  # restore original reading order
    return " ".join(sentences[i] for i in ranked_idx)


# --------------------------------------------------------------------------
# Keyword extraction
# --------------------------------------------------------------------------

def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """YAKE is unsupervised and statistical (no pretrained model to
    download), which keeps this fast and deploy-friendly. Returns
    keywords/short phrases ranked by relevance, best first.
    """
    if not text or not text.strip():
        return []

    extractor = yake.KeywordExtractor(
        lan="en",
        n=2,  # allow up to 2-word phrases
        dedupLim=0.7,
        top=max_keywords,
    )
    ranked = extractor.extract_keywords(text)
    # yake returns (keyword, score) with LOWER score = more relevant
    ranked.sort(key=lambda pair: pair[1])
    return [kw for kw, _score in ranked][:max_keywords]


# --------------------------------------------------------------------------
# Orchestration / caching
# --------------------------------------------------------------------------

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
