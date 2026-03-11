"""
Main Pipeline Orchestrator — Mac M2 Optimized
Runs every 6 hours via scheduler.
Coordinates: Scrape → Match → Tailor → Apply → Track
"""

import time
import json
import psutil
import schedule
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from scraper.job_scraper import run_scraper, get_unprocessed_jobs, update_job_status
from matcher.similarity_engine import filter_jobs_by_similarity
from llm.resume_optimizer import batch_tailor_top_jobs
from automation.auto_apply import run_auto_apply, get_application_stats

import asyncio

# ─── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "database" / "applications.db"
LOG_FILE = Path(__file__).parent / "database" / "pipeline_log.json"

PIPELINE_INTERVAL_HOURS = 6       # Run every 6 hours
MAX_CPU_PERCENT = 60              # Pause if CPU exceeds this
MAX_JOBS_TO_TAILOR = 10           # LLM calls per run
SIMILARITY_THRESHOLD = 0.60      # Jobs below this are skipped

# ─── CPU Guard ─────────────────────────────────────────────────────────────────

def wait_for_cpu_cooldown(max_cpu: float = MAX_CPU_PERCENT, interval: int = 5):
    """Wait until CPU usage drops below threshold before proceeding."""
    cpu = psutil.cpu_percent(interval=1)
    if cpu > max_cpu:
        print(f"⚠️  CPU at {cpu:.0f}% (>{max_cpu}%). Waiting for cooldown...")
        while cpu > max_cpu:
            time.sleep(interval)
            cpu = psutil.cpu_percent(interval=1)
            print(f"  CPU: {cpu:.0f}%")
        print(f"✅ CPU cooled down to {cpu:.0f}%")


# ─── Pipeline Steps ────────────────────────────────────────────────────────────

def step_scrape() -> list[dict]:
    """Step 1: Scrape new job listings."""
    print("\n" + "="*60)
    print("📡 STEP 1: SCRAPING JOBS")
    print("="*60)
    wait_for_cpu_cooldown()

    try:
        asyncio.run(run_scraper())
        jobs = get_unprocessed_jobs(limit=100)
        print(f"✅ Retrieved {len(jobs)} unprocessed jobs from database")
        return jobs
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return []


def step_match(jobs: list[dict]) -> list[dict]:
    """Step 2: Filter jobs by semantic similarity (no LLM needed)."""
    print("\n" + "="*60)
    print("🔍 STEP 2: SIMILARITY MATCHING (FAISS)")
    print("="*60)
    wait_for_cpu_cooldown()

    if not jobs:
        print("No jobs to match.")
        return []

    try:
        matched = filter_jobs_by_similarity(jobs, threshold=SIMILARITY_THRESHOLD)

        # Update DB status for non-matched jobs
        matched_ids = {j.get("id") for j in matched}
        for job in jobs:
            if job.get("id") not in matched_ids:
                update_job_status(
                    job["id"],
                    status="low_match",
                    score=job.get("similarity_score", 0.0)
                )

        return matched

    except Exception as e:
        print(f"❌ Matching failed: {e}")
        return []


def step_tailor(matched_jobs: list[dict]) -> list[dict]:
    """Step 3: Use LLM to tailor resumes for top matched jobs."""
    print("\n" + "="*60)
    print("✍️  STEP 3: RESUME TAILORING (LLM)")
    print("="*60)
    wait_for_cpu_cooldown()

    if not matched_jobs:
        print("No matched jobs to tailor.")
        return []

    try:
        tailored = batch_tailor_top_jobs(matched_jobs, max_jobs=MAX_JOBS_TO_TAILOR)

        # Update DB with tailored results
        for job in tailored:
            if job.get("tailored_resume"):
                update_job_status(
                    job["id"],
                    status="tailored",
                    score=job.get("similarity_score", 0.0),
                    tailored_resume=job.get("tailored_resume", ""),
                    cover_letter=job.get("cover_letter", ""),
                )

        return tailored

    except Exception as e:
        print(f"❌ Tailoring failed: {e}")
        return matched_jobs  # Return untailored for manual review


def step_apply(tailored_jobs: list[dict]):
    """Step 4: Auto-apply to tailored jobs."""
    print("\n" + "="*60)
    print("🤖 STEP 4: AUTO APPLYING")
    print("="*60)
    wait_for_cpu_cooldown()

    if not tailored_jobs:
        print("No tailored jobs ready for application.")
        return

    try:
        asyncio.run(run_auto_apply(tailored_jobs))
    except Exception as e:
        print(f"❌ Auto-apply failed: {e}")


def step_report():
    """Step 5: Print pipeline summary."""
    print("\n" + "="*60)
    print("📊 STEP 5: PIPELINE SUMMARY")
    print("="*60)

    stats = get_application_stats()
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()

    print(f"\n  📬 Applications today:  {stats['today']}/5")
    print(f"  📬 Total applications:  {stats['total']}")
    print(f"  ⏳ Jobs pending apply:  {stats['pending']}")
    print(f"  💻 CPU usage:           {cpu:.0f}%")
    print(f"  🧠 RAM usage:           {mem.percent:.0f}%")
    print(f"  🕐 Next run:            in {PIPELINE_INTERVAL_HOURS} hours")

    log_pipeline_run(stats, cpu, mem.percent)


# ─── Logging ───────────────────────────────────────────────────────────────────

def log_pipeline_run(stats: dict, cpu: float, ram: float):
    """Append pipeline run summary to log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []

    logs.append({
        "timestamp": datetime.now().isoformat(),
        "applied_today": stats["today"],
        "total_applied": stats["total"],
        "pending": stats["pending"],
        "cpu_percent": cpu,
        "ram_percent": ram,
    })

    # Keep only last 100 runs
    logs = logs[-100:]

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


# ─── Full Pipeline Run ─────────────────────────────────────────────────────────

def run_full_pipeline():
    """Execute the complete pipeline end-to-end."""
    start_time = datetime.now()
    print(f"\n🚀 PIPELINE STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mac M2 Optimized AI Job Agent")
    print("=" * 60)

    try:
        # Step 1: Scrape
        jobs = step_scrape()

        # --- Recovery: Rescore any 'new' jobs that missed scoring ---
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE status = 'new' AND similarity_score = 0.0")
        missed_scoring = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if missed_scoring:
            print(f"🔄 Rescoring {len(missed_scoring)} jobs that missed initial matching...")
            jobs.extend(missed_scoring)

        # Step 2: Match (FAISS — no LLM)
        matched = step_match(jobs)

        # Step 3: Tailor (LLM — only for matched jobs)
        # We only tailor jobs that are matched but not already tailored
        untailored = [j for j in matched if j.get("status") != "tailored"]
        step_tailor(untailored)

        # Step 4: Apply
        # IMPORTANT: We fetch ALL tailored jobs from the DB to ensure nothing is missed
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE status = 'tailored' AND applied = 0")
        all_tailored = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if all_tailored:
            print(f"📋 Found {len(all_tailored)} tailored jobs ready to apply.")
            step_apply(all_tailored)
        else:
            print("ℹ️  No tailored jobs ready for application.")

        # Step 5: Report
        step_report()

    except KeyboardInterrupt:
        print("\n\n⛔ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n✅ Pipeline completed in {elapsed:.1f} minutes")


# ─── Scheduler ─────────────────────────────────────────────────────────────────

def start_scheduler():
    """Run pipeline every 6 hours using schedule library."""
    print(f"⏰ Scheduler started — pipeline runs every {PIPELINE_INTERVAL_HOURS} hours")
    print("   Press Ctrl+C to stop\n")

    # Run immediately on startup
    run_full_pipeline()

    # Then schedule recurring runs
    schedule.every(PIPELINE_INTERVAL_HOURS).hours.do(run_full_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def run_once():
    """Run the pipeline exactly once (for testing)."""
    run_full_pipeline()


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once()
    else:
        start_scheduler()
