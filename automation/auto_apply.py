"""
Auto Apply Bot — Updated to use LinkedIn Session Manager
Uses saved cookies for headless operation — no repeated logins.
"""

import asyncio
import json
import random
import sqlite3
from datetime import datetime, date
from pathlib import Path
from playwright.async_api import async_playwright, Page

# Import session manager and external handler
from automation.linkedin_session import get_authenticated_context, is_session_valid
from automation.external_handler import handle_external_site

DB_PATH = Path(__file__).parent.parent / "database" / "applications.db"
LOG_PATH = Path(__file__).parent.parent / "database" / "apply_log.json"

# Safety limits
MAX_APPLY_PER_DAY = 5
DELAY_BETWEEN_APPLY = (10, 20)  # seconds


async def human_type(page: Page, selector: str, text: str):
    """Type text character by character to simulate human input."""
    element = await page.wait_for_selector(selector, timeout=10000)
    await element.click()
    await asyncio.sleep(random.uniform(0.3, 0.7))
    for char in text:
        await element.type(char, delay=random.uniform(30, 120))
    await asyncio.sleep(random.uniform(0.3, 0.8))


async def random_sleep(min_s: float = 10, max_s: float = 20):
    delay = random.uniform(min_s, max_s)
    print(f"  ⏳ Waiting {delay:.1f}s...")
    await asyncio.sleep(delay)


def get_today_apply_count() -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = date.today().isoformat()
    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied = 1 AND date(applied_at) = ?",
        (today,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def mark_as_applied(job_id: int, status: str = "applied"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE jobs SET applied = 1, applied_at = ?, status = ? WHERE id = ?",
        (datetime.now().isoformat(), status, job_id)
    )
    conn.commit()
    conn.close()


def log_application(job: dict, success: bool, error: str = ""):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "job_id": job.get("id"),
        "title": job.get("title"),
        "company": job.get("company"),
        "url": job.get("url"),
        "source": job.get("source"),
        "similarity_score": job.get("similarity_score"),
        "success": success,
        "error": error,
    })
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


async def handle_easy_apply_modal(page: Page, job: dict) -> bool:
    """
    Handle the LinkedIn Easy Apply multi-step modal.
    Smart enough to fill phone/contact fields if needed.
    """
    from resume.resume import load_resume_data  # loaded lazily

    max_steps = 8
    for step in range(max_steps):
        print(f"  📋 Easy Apply step {step + 1}...")
        await asyncio.sleep(1.5)

        # ── Submit button ─────────────────────────────────────────────
        submit_btn = await page.query_selector(
            "button[aria-label='Submit application'],"
            "button:has-text('Submit application')"
        )
        if submit_btn:
            print("  📨 Submitting application...")
            await submit_btn.click()
            await asyncio.sleep(3)

            # Success confirmation
            success_modal = await page.query_selector(
                ".jobs-easy-apply-modal__post-apply-content,"
                "[data-test-modal='easy-apply-success'],"
                ".artdeco-toaster__message"
            )
            if success_modal:
                print("  🎉 Application submitted successfully!")
                return True
            return True  # Assume success if no error shown

        # ── Review button ─────────────────────────────────────────────
        review_btn = await page.query_selector(
            "button[aria-label='Review your application'],"
            "button:has-text('Review')"
        )
        if review_btn:
            await review_btn.click()
            continue

        # ── Handle "Resume" step — upload or use existing ─────────────
        resume_section = await page.query_selector(
            "[data-test-resume-upload-option], .jobs-resume-picker"
        )
        if resume_section:
            if job.get("resume_pdf") and Path(job["resume_pdf"]).exists():
                print(f"  ⬆️  Uploading customized resume PDF: {Path(job['resume_pdf']).name}")
                file_input = await page.query_selector("input[type='file']")
                if file_input:
                    try:
                        await file_input.set_input_files(job["resume_pdf"])
                        await asyncio.sleep(2)
                    except Exception as e:
                        print(f"  ⚠️  Failed to upload PDF: {e}")
            else:
                print("  📄 No custom PDF found. Using existing LinkedIn profile resume")
                existing_resume = await page.query_selector(
                    "input[type='radio'][aria-label*='resume'], "
                    ".jobs-resume-picker__resume-btn"
                )
                if existing_resume:
                    await existing_resume.click()
                    await asyncio.sleep(1)

        # ── Handle phone number field ──────────────────────────────────
        phone_input = await page.query_selector(
            "input[id*='phone'], input[name*='phone'], "
            "input[aria-label*='Phone number']"
        )
        if phone_input:
            current_val = await phone_input.input_value()
            if not current_val:
                print("  📞 Filling phone number...")
                await phone_input.fill("+919359768168")
                await asyncio.sleep(0.5)

        # ── Handle "Years of experience" dropdowns ────────────────────
        selects = await page.query_selector_all("select")
        for sel in selects:
            selected = await sel.input_value()
            if not selected or selected == "":
                # Pick first non-empty option
                options = await sel.query_selector_all("option")
                if len(options) > 1:
                    val = await options[1].get_attribute("value")
                    if val:
                        await sel.select_option(value=val)
                await asyncio.sleep(0.3)

        # ── Handle text areas (cover letter / additional info) ─────────
        textareas = await page.query_selector_all("textarea")
        for ta in textareas:
            current = await ta.input_value()
            if not current and job.get("cover_letter"):
                # Use first 500 chars of cover letter
                await ta.fill(job["cover_letter"][:500])
                await asyncio.sleep(0.5)

        # ── Next button ───────────────────────────────────────────────
        next_btn = await page.query_selector(
            "button[aria-label='Continue to next step'],"
            "button:has-text('Next'),"
            "button[data-easy-apply-next-button]"
        )
        if next_btn:
            await next_btn.click()
            await asyncio.sleep(2)
        else:
            print(f"  ⚠️  No action button found at step {step + 1}. Stopping.")
            break

    return False


async def apply_to_job(page: Page, job: dict) -> bool:
    """Apply to a single LinkedIn job using Easy Apply."""
    try:
        print(f"  🌐 Opening: {job['url'][:80]}...")
        await page.goto(job["url"], wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(random.uniform(2, 4))

        # ── Find Easy Apply button ─────────────────────────────────────
        easy_apply_btn = await page.query_selector(
            ".jobs-apply-button--top-card button,"
            "button.jobs-apply-button,"
            "[data-control-name='jobdetails_topcard_inapply']"
        )

        if not easy_apply_btn:
            print("  ⚠️  No Easy Apply button. Attempting to apply on company site...")
            from resume.resume import load_resume_data
            resume_data = load_resume_data()
            success = await handle_external_site(page.context, page, job, resume_data)
            return success

        btn_text = await easy_apply_btn.inner_text()
        if "Easy Apply" not in btn_text:
            print(f"  ℹ️  Not Easy Apply ({btn_text.strip()[:30]}). Attempting external application...")
            from resume.resume import load_resume_data
            resume_data = load_resume_data()
            success = await handle_external_site(page.context, page, job, resume_data)
            return success

        print("  🖱️  Clicking Easy Apply...")
        await easy_apply_btn.click()
        await asyncio.sleep(2)

        # ── Handle modal ──────────────────────────────────────────────
        success = await handle_easy_apply_modal(page, job)
        return success

    except Exception as e:
        print(f"  ❌ Apply error: {e}")
        return False


async def save_for_manual_review(job: dict):
    """Save job to manual review folder."""
    review_dir = Path(__file__).parent.parent / "database" / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    safe_name = (
        f"{job.get('company', 'unknown')}_{job.get('title', 'unknown')}"
        .replace(" ", "_")[:40]
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = review_dir / f"{ts}_{safe_name}.json"

    with open(out, "w") as f:
        json.dump({
            "job": {k: v for k, v in job.items() if k != "tailored_resume"},
            "tailored_resume": job.get("tailored_resume", ""),
            "cover_letter": job.get("cover_letter", ""),
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  📁 Saved for manual review: {out.name}")


async def run_auto_apply(jobs: list[dict]):
    """Main auto-apply pipeline — uses LinkedIn session manager."""
    print("\n🤖 Starting Auto Apply Bot")
    print("=" * 50)

    # ── Check session ─────────────────────────────────────────────────
    if not is_session_valid():
        print("\n❌ No valid LinkedIn session!")
        print("   Run this first:")
        print("   python automation/linkedin_session.py --login")
        return

    # ── Daily limit ───────────────────────────────────────────────────
    today_count = get_today_apply_count()
    remaining = MAX_APPLY_PER_DAY - today_count

    if remaining <= 0:
        print(f"  🛑 Daily limit reached ({MAX_APPLY_PER_DAY}/day). Stopping.")
        return

    print(f"  📊 Applied today: {today_count}/{MAX_APPLY_PER_DAY} | Remaining: {remaining}")

    # ── Filter ready jobs ─────────────────────────────────────────────
    ready_jobs = [j for j in jobs if j.get("tailored_resume") and j.get("url")]
    ready_jobs = sorted(ready_jobs, key=lambda j: j.get("similarity_score", 0), reverse=True)
    to_apply = ready_jobs[:remaining]

    if not to_apply:
        print("  ℹ️  No jobs ready (need tailored resumes first)")
        return

    print(f"  📋 Attempting {len(to_apply)} applications")

    applied_count = 0

    async with async_playwright() as p:
        try:
            browser, context, page = await get_authenticated_context(p)
        except RuntimeError as e:
            print(f"  ❌ Session error: {e}")
            return

        try:
            for i, job in enumerate(to_apply):
                print(f"\n[{i+1}/{len(to_apply)}] {job.get('title')} @ {job.get('company')}")
                print(f"  📊 Match: {job.get('similarity_score', 0):.0%}")

                result = await apply_to_job(page, job)

                if result is True:
                    applied_count += 1
                    mark_as_applied(job["id"])
                    log_application(job, success=True)
                    print(f"  ✅ Application complete! (total today: {applied_count})")

                    if applied_count >= MAX_APPLY_PER_DAY:
                        print(f"\n🛑 Daily limit {MAX_APPLY_PER_DAY} reached. Stopping.")
                        break

                elif result is None:
                    # This branch is now largely unreachable as we try external apply
                    await save_for_manual_review(job)
                    log_application(job, success=False, error="Job unreachable or skipped")

                else:
                    # Failed
                    log_application(job, success=False, error="Application attempt failed")
                    await save_for_manual_review(job)

                if i < len(to_apply) - 1:
                    await random_sleep(*DELAY_BETWEEN_APPLY)

        finally:
            await browser.close()

    print(f"\n✅ Done! Applied to {applied_count} job(s) today.")


def get_application_stats() -> dict:
    if not DB_PATH.exists():
        return {"total": 0, "today": 0, "pending": 0}
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE applied = 1")
    total = cursor.fetchone()[0]
    today = date.today().isoformat()
    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied = 1 AND date(applied_at) = ?", (today,)
    )
    today_count = cursor.fetchone()[0]
    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied = 0 AND similarity_score >= 0.6"
    )
    pending = cursor.fetchone()[0]
    conn.close()
    return {"total": total, "today": today_count, "pending": pending}


if __name__ == "__main__":
    stats = get_application_stats()
    print("📊 Application Stats:")
    print(f"  Total applied: {stats['total']}")
    print(f"  Applied today: {stats['today']}/{MAX_APPLY_PER_DAY}")
    print(f"  Pending:       {stats['pending']}")
