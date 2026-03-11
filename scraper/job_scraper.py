"""
Job Scraper — Mac M2 Optimized
Uses Playwright in headless mode for low CPU usage.
Scrapes LinkedIn, Indeed, and custom job boards.
"""

import asyncio
import json
import sqlite3
import random
import time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

DB_PATH = Path(__file__).parent.parent / "database" / "applications.db"
JOBS_CACHE = Path(__file__).parent.parent / "database" / "scraped_jobs.json"

TARGET_KEYWORDS = [
    "AI Engineer",
    "Machine Learning Engineer",
    "LLM Engineer",
    "RAG Engineer",
    "Python AI",
    "Backend AI Engineer",
    "NLP Engineer",
]

TARGET_LOCATIONS = ["Remote", "India", "Chandigarh"]

MAX_JOBS_PER_RUN = 100


async def random_sleep(min_s=10, max_s=20):
    """Polite random delay to avoid detection and CPU spikes."""
    delay = random.uniform(min_s, max_s)
    print(f"  ⏳ Sleeping {delay:.1f}s...")
    await asyncio.sleep(delay)


async def scrape_linkedin_jobs(page, keyword: str, location: str) -> list[dict]:
    """Scrape LinkedIn job listings for a given keyword and location."""
    jobs = []

    search_url = (
        f"https://www.linkedin.com/jobs/search/"
        f"?keywords={keyword.replace(' ', '%20')}"
        f"&location={location.replace(' ', '%20')}"
        f"&f_TPR=r86400"  # last 24 hours
        f"&sortBy=DD"
    )

    try:
        print(f"  🔍 LinkedIn: {keyword} @ {location}")
        await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        # Scroll to load more jobs
        for _ in range(3):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

        # Extract job cards
        job_cards = await page.query_selector_all(".job-search-card")

        for card in job_cards[:20]:
            try:
                title_el = await card.query_selector(".base-search-card__title")
                company_el = await card.query_selector(".base-search-card__subtitle")
                location_el = await card.query_selector(".job-search-card__location")
                link_el = await card.query_selector("a.base-card__full-link")

                title = await title_el.inner_text() if title_el else ""
                company = await company_el.inner_text() if company_el else ""
                loc = await location_el.inner_text() if location_el else ""
                link = await link_el.get_attribute("href") if link_el else ""

                if title and company:
                    jobs.append({
                        "source": "linkedin",
                        "title": title.strip(),
                        "company": company.strip(),
                        "location": loc.strip(),
                        "url": link.strip(),
                        "keyword": keyword,
                        "description": "",
                        "scraped_at": datetime.now().isoformat(),
                        "applied": False,
                        "similarity_score": 0.0,
                    })
            except Exception as e:
                print(f"    ⚠️  Card parse error: {e}")
                continue

        print(f"  ✅ Found {len(jobs)} jobs from LinkedIn")

    except Exception as e:
        print(f"  ❌ LinkedIn scrape failed: {e}")

    return jobs


async def fetch_job_description(page, url: str) -> str:
    """Fetch full job description from job detail page."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(2)

        # LinkedIn description
        desc_el = await page.query_selector(".show-more-less-html__markup")
        if desc_el:
            return (await desc_el.inner_text()).strip()

        # Generic fallback — grab main content
        body = await page.query_selector("main")
        if body:
            text = await body.inner_text()
            return text[:3000].strip()  # Limit to 3000 chars

    except Exception as e:
        print(f"    ⚠️  Description fetch failed: {e}")

    return ""


def normalize_url(url: str) -> str:
    """Remove tracking parameters from LinkedIn URLs."""
    if not url: return ""
    return url.split('?')[0]


def save_jobs_to_db(jobs: list[dict]):
    """Save scraped jobs to SQLite database with deduplication."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            company TEXT,
            location TEXT,
            url TEXT UNIQUE,
            keyword TEXT,
            description TEXT,
            scraped_at TEXT,
            applied INTEGER DEFAULT 0,
            similarity_score REAL DEFAULT 0.0,
            tailored_resume TEXT,
            cover_letter TEXT,
            applied_at TEXT,
            status TEXT DEFAULT 'new'
        )
    """)

    # Get existing titles/companies to avoid duplicates even with different URLs
    cursor.execute("SELECT title, company FROM jobs")
    existing = {(row[0].lower(), row[1].lower()) for row in cursor.fetchall()}

    inserted = 0
    seen_in_batch = set()

    for job in jobs:
        # Normalize
        job['url'] = normalize_url(job.get('url', ''))
        title_key = (job['title'].lower(), job['company'].lower())

        if title_key in existing or title_key in seen_in_batch:
            continue
            
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO jobs
                (source, title, company, location, url, keyword, description,
                 scraped_at, applied, similarity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job["source"], job["title"], job["company"],
                job["location"], job["url"], job["keyword"],
                job["description"], job["scraped_at"],
                int(job["applied"]), job["similarity_score"]
            ))
            if cursor.rowcount > 0:
                inserted += 1
                seen_in_batch.add(title_key)
        except Exception as e:
            print(f"  ⚠️  DB insert error: {e}")

    conn.commit()
    conn.close()
    print(f"  💾 Saved {inserted} new unique jobs to database")


async def run_scraper():
    """Main scraper entry point."""
    print("\n🚀 Starting Job Scraper (Mac M2 Optimized)")
    print("=" * 50)

    all_jobs = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,  # Low CPU mode
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",   # Use CPU rendering only
                "--disable-extensions",
            ]
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )

        page = await context.new_page()

        # Scrape each keyword × location (batched to avoid overload)
        for keyword in TARGET_KEYWORDS[:3]:  # Limit to 3 keywords per run
            for location in TARGET_LOCATIONS[:2]:  # Limit to 2 locations
                jobs = await scrape_linkedin_jobs(page, keyword, location)
                all_jobs.extend(jobs)
                await random_sleep(5, 10)  # Polite delay between searches

                if len(all_jobs) >= MAX_JOBS_PER_RUN:
                    print(f"  📊 Reached max jobs limit ({MAX_JOBS_PER_RUN})")
                    break

        # Fetch descriptions for top jobs only (avoid overloading CPU)
        print(f"\n📝 Fetching descriptions for top {min(10, len(all_jobs))} jobs...")
        for i, job in enumerate(all_jobs[:10]):
            if job["url"]:
                print(f"  [{i+1}/10] {job['title']} @ {job['company']}")
                job["description"] = await fetch_job_description(page, job["url"])
                await random_sleep(3, 7)

        await browser.close()

    # Save results
    save_jobs_to_db(all_jobs)

    # Also save to JSON cache for inspection
    JOBS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS_CACHE, "w") as f:
        json.dump(all_jobs, f, indent=2)

    print(f"\n✅ Scraping complete! Total jobs found: {len(all_jobs)}")
    return all_jobs


def get_unprocessed_jobs(limit: int = 50) -> list[dict]:
    """Retrieve unprocessed jobs from the database."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, source, title, company, location, url, description,
               keyword, scraped_at
        FROM jobs
        WHERE status = 'new' AND applied = 0
        ORDER BY scraped_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    jobs = []
    for row in rows:
        jobs.append({
            "id": row[0],
            "source": row[1],
            "title": row[2],
            "company": row[3],
            "location": row[4],
            "url": row[5],
            "description": row[6],
            "keyword": row[7],
            "scraped_at": row[8],
        })

    return jobs


def update_job_status(job_id: int, status: str, score: float = 0.0,
                       tailored_resume: str = "", cover_letter: str = ""):
    """Update job status in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE jobs
        SET status = ?, similarity_score = ?, tailored_resume = ?,
            cover_letter = ?
        WHERE id = ?
    """, (status, score, tailored_resume, cover_letter, job_id))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    asyncio.run(run_scraper())
