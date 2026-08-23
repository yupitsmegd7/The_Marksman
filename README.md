# The MarksMan

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/marksman/blob/main/LICENSE)

## Overview

**Marksman** is a lightweight news‑aggregation web application that pulls headlines from multiple RSS feeds, Google News, and custom sources into a single, clean UI. It is designed for professionals who want to stay up‑to‑date on niche topics that matter to their career (e.g., exams, education news, scholarships, KIIT campus updates, Bengal regional news, etc.).

The goal is to bring relevant headlines together in one place so you don't have to wander across many sites.

---

## Live Demo

[https://github.com/yourusername/marksman](https://github.com/yourusername/marksman) <!-- replace with actual repository URL -->

---

## Services Provided

- **RSS Feed Ingestion** – Pulls articles from a curated list of RSS sources (Indian Express, ABP Live, NDTV, etc.).
- **Google News Integration** – Uses the Google News API to fetch additional headlines for each category.
- **Custom Bengal Section** – Aggregates regional news from *Ei Samay* and other Bengali outlets.
- **Image Extraction & Back‑filling** – Retrieves article images via RSS media tags, OpenGraph (`og:image`) scraping, and a back‑fill routine for older items.
- **SQLite Persistence** – Stores articles, hashes, and images in `news_portal.db` for fast retrieval.
- **FastAPI Backend** – Provides REST endpoints (`/news/{category}`) consumed by the frontend.
- **Responsive Frontend** – Built with modern HTML/CSS (grid layout, CSS variables) and vanilla JavaScript. No heavy frameworks required.

---

## How It Works

1. **Configuration** – `app.py` defines a `CATEGORIES` dictionary mapping category names to a list of RSS URLs and Google News queries.
2. **Ingestion Pipeline** – `run_ingestion()` (called periodically) iterates over each category, fetches the RSS feed, extracts title, link, description, and image URL. If an image is missing, the pipeline attempts to scrape the article’s OpenGraph tags or fallback to a placeholder.
3. **Deduplication** – Articles are identified by a SHA‑256 hash of the URL (`url_hash`). New items are inserted; existing items are updated only when a missing image is discovered.
4. **Database** – SQLAlchemy models (`Article`) persist data in `news_portal.db`.
5. **API** – FastAPI exposes endpoints like:
   ```
   GET /news/{category}?limit=20
   ```
   which return JSON of the latest articles for that category.
6. **Frontend** – The client periodically requests the API, renders cards with headline, image, and short description, and offers a responsive grid that works on desktop and mobile.

---

## Data Sources

| Category | Primary RSS Feeds | Google News Query |
|----------|-------------------|-------------------|
| Exams | `https://indianexpress.com/section/exams/feed/` | `exams` |
| Education News | `https://indianexpress.com/section/education/feed/`, `https://news.abplive.com/education/feed` | `education` |
| KIIT News | `https://news.kiit.ac.in/feed/` (WordPress) | `kiit university` |
| Bengal News | `https://eisamay.com/rss` and other Bengali outlets | `bengal news` |
| Scholarships, Campus Events, Student News, etc. – similar RSS + Google News sources |

All feeds are public and do not require API keys.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/marksman.git
cd marksman

# Set up a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser to see the aggregator.

---

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for:
- Adding new RSS sources.
- Improving image extraction.
- Enhancing UI/UX.
- Fixing bugs.

Please follow the Contributor Covenant Code of Conduct.

---

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.
