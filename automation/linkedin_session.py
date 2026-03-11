"""
LinkedIn Session Manager
Handles login, session persistence, and cookie reuse.
Login is done ONCE manually — then cookies are saved and reused automatically.
This means the bot never has to log in again until cookies expire (~30 days).
"""

import asyncio
import json
import getpass
import sys
from pathlib import Path
from datetime import datetime, timedelta
from playwright.async_api import async_playwright, BrowserContext, Page

SESSION_DIR = Path(__file__).parent.parent / "database" / "sessions"
COOKIES_FILE = SESSION_DIR / "linkedin_cookies.json"
SESSION_META_FILE = SESSION_DIR / "session_meta.json"

SESSION_VALIDITY_DAYS = 25  # Refresh before 30-day LinkedIn expiry
LINKEDIN_HOME = "https://www.linkedin.com"
LINKEDIN_FEED = "https://www.linkedin.com/feed"
LINKEDIN_LOGIN_URL = "https://www.linkedin.com/login"


# ─── Session Health ────────────────────────────────────────────────────────────

def is_session_valid() -> bool:
    """Check if a saved LinkedIn session exists and is not expired."""
    if not COOKIES_FILE.exists() or not SESSION_META_FILE.exists():
        return False

    with open(SESSION_META_FILE) as f:
        meta = json.load(f)

    saved_at = datetime.fromisoformat(meta.get("saved_at", "2000-01-01"))
    expiry = saved_at + timedelta(days=SESSION_VALIDITY_DAYS)

    if datetime.now() > expiry:
        print("  ⚠️  LinkedIn session expired. Re-login required.")
        return False

    days_remaining = (expiry - datetime.now()).days
    print(f"  ✅ Valid LinkedIn session found (expires in {days_remaining} days)")
    return True


def get_session_info() -> dict:
    """Return metadata about the current saved session."""
    if not SESSION_META_FILE.exists():
        return {"status": "no_session"}

    with open(SESSION_META_FILE) as f:
        return json.load(f)


def clear_session():
    """Delete saved session (force re-login on next run)."""
    if COOKIES_FILE.exists():
        COOKIES_FILE.unlink()
    if SESSION_META_FILE.exists():
        SESSION_META_FILE.unlink()
    print("  🗑️  Session cleared. Re-login required on next run.")


# ─── Save / Load Cookies ───────────────────────────────────────────────────────

def save_cookies(cookies: list, email: str):
    """Persist browser cookies to disk."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    with open(COOKIES_FILE, "w") as f:
        json.dump(cookies, f, indent=2)

    with open(SESSION_META_FILE, "w") as f:
        json.dump({
            "email": email,
            "saved_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=SESSION_VALIDITY_DAYS)).isoformat(),
            "cookie_count": len(cookies),
        }, f, indent=2)

    print(f"  💾 Session saved ({len(cookies)} cookies)")


async def load_cookies_into_context(context: BrowserContext) -> bool:
    """Load saved cookies into a Playwright browser context."""
    if not COOKIES_FILE.exists():
        return False

    with open(COOKIES_FILE) as f:
        cookies = json.load(f)

    if not cookies:
        return False

    await context.add_cookies(cookies)
    print(f"  🍪 Loaded {len(cookies)} cookies into browser")
    return True


# ─── Login Flow ────────────────────────────────────────────────────────────────

async def verify_logged_in(page: Page) -> bool:
    """Check if the current page state indicates we're logged in."""
    try:
        # Navigate to feed and wait for basic content
        print("  ⏳ Checking LinkedIn feed...")
        await page.goto(LINKEDIN_FEED, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)  # More time for dynamic content

        current_url = page.url
        title = await page.title()

        # ── Fast-path check ───────────────────────────────────────────
        # If we're on the feed page or have 'Feed' in the title, we're in
        if "feed" in current_url.lower() or "feed" in title.lower():
            print(f"  ✅ Verified by URL/Title: {title}")
            return True

        # If redirected to login/checkpoint, we're definitely out
        if "login" in current_url or "checkpoint" in current_url:
            print(f"  ❌ Redirected to login: {current_url}")
            return False

        # ── Fallback: DOM Selectors ──────────────────────────────────
        nav = await page.query_selector("nav.global-nav, .global-nav__me-photo")
        if nav:
            print("  ✅ Verified by DOM element")
            return True

        print(f"  ⚠️  Unknown state. URL: {current_url} | Title: {title}")
        return False

    except Exception as e:
        print(f"  ⚠️  Login check error: {e}")
        return False


async def perform_login(page: Page, email: str, password: str) -> bool:
    """
    Perform LinkedIn login with human-like behavior.
    Handles 2FA checkpoint detection.
    """
    import random

    print(f"  🔐 Logging in as {email}...")

    try:
        await page.goto(LINKEDIN_LOGIN_URL, wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(random.uniform(1.5, 3))

        # Fill email
        email_input = await page.wait_for_selector("#username", timeout=10000)
        await email_input.click()
        await asyncio.sleep(random.uniform(0.3, 0.7))

        # Human-like typing
        for char in email:
            await email_input.type(char, delay=random.uniform(40, 120))

        await asyncio.sleep(random.uniform(0.5, 1.2))

        # Fill password
        pass_input = await page.wait_for_selector("#password", timeout=5000)
        await pass_input.click()
        await asyncio.sleep(random.uniform(0.3, 0.7))

        for char in password:
            await pass_input.type(char, delay=random.uniform(40, 120))

        await asyncio.sleep(random.uniform(0.8, 1.5))

        # Click submit
        submit_btn = await page.wait_for_selector(
            "button[type='submit'], button[data-litms-control-urn='login-submit']",
            timeout=5000
        )
        await submit_btn.click()
        print("  ⏳ Waiting for login response...")

        # Wait for navigation
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
        await asyncio.sleep(3)

        current_url = page.url

        # ── Handle 2FA / CAPTCHA checkpoint ────────────────────────────
        if "checkpoint" in current_url or "challenge" in current_url:
            print("\n  ⚠️  LinkedIn Security Check Detected!")
            print("  ➡️  Please complete the verification in the browser window.")
            print("  ➡️  You have 60 seconds...")

            # Wait for manual completion
            timeout = 60
            for i in range(timeout):
                await asyncio.sleep(1)
                url_now = page.url
                if "checkpoint" not in url_now and "challenge" not in url_now:
                    print("  ✅ Checkpoint passed!")
                    break
                if i == timeout - 1:
                    print("  ❌ Checkpoint timeout. Login failed.")
                    return False

        # ── Handle Phone Verification ─────────────────────────────────
        if "add-phone" in current_url or "phone" in current_url:
            print("  ℹ️  LinkedIn asking for phone — skipping...")
            skip_btn = await page.query_selector("button:has-text('Skip'), a:has-text('Skip')")
            if skip_btn:
                await skip_btn.click()
                await asyncio.sleep(2)

        # Final check
        await page.goto(LINKEDIN_FEED, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(2)

        if "feed" in page.url or "in/" in page.url:
            print("  ✅ Login successful!")
            return True

        print(f"  ❌ Login failed. Current URL: {page.url}")
        return False

    except Exception as e:
        print(f"  ❌ Login error: {e}")
        return False


# ─── Main Session Creator ──────────────────────────────────────────────────────

async def create_linkedin_session(email: str = None, password: str = None):
    """
    Interactive LinkedIn login + session save.
    Run this once to set up your session.
    Will open a real (visible) browser window.
    """
    print("\n🔐 LinkedIn Session Setup")
    print("=" * 50)
    print("This will open a browser window for you to log in.")
    print("Your session will be saved and reused automatically.\n")

    if not email:
        email = input("LinkedIn email: ").strip()
    if not password:
        password = getpass.getpass("LinkedIn password: ")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # MUST be visible for login
            slow_mo=100,
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        page = await context.new_page()

        success = await perform_login(page, email, password)

        if success:
            # Save session cookies
            cookies = await context.cookies()
            save_cookies(cookies, email)

            print("\n✅ Session created successfully!")
            print("   Future runs will use saved cookies (no re-login needed)")
        else:
            print("\n❌ Login failed. Please try again.")

        await asyncio.sleep(2)
        await browser.close()

    return success


async def get_authenticated_context(playwright) -> tuple:
    """
    Return an authenticated browser context.
    Uses saved cookies if valid, otherwise raises an error.
    
    Returns: (browser, context, page)
    """
    if not is_session_valid():
        raise RuntimeError(
            "No valid LinkedIn session found.\n"
            "Run: python automation/linkedin_session.py --login"
        )

    browser = await playwright.chromium.launch(
        headless=False,  # Set to False so LinkedIn doesn't detect it as a bot
        args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
        ]
    )

    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
    )

    # Load saved cookies
    loaded = await load_cookies_into_context(context)
    if not loaded:
        await browser.close()
        raise RuntimeError("Failed to load session cookies")

    page = await context.new_page()

    # Verify we're still logged in
    print("  🔍 Verifying LinkedIn session...")
    logged_in = await verify_logged_in(page)

    if not logged_in:
        await browser.close()
        raise RuntimeError(
            "Session cookies are no longer valid (LinkedIn may have expired them).\n"
            "Run: python automation/linkedin_session.py --login"
        )

    print("  ✅ LinkedIn session verified!")
    return browser, context, page


async def refresh_session_if_needed():
    """
    Check session validity and auto-refresh cookies by navigating LinkedIn.
    Call this at the start of each pipeline run.
    """
    if not is_session_valid():
        return False

    print("  🔄 Refreshing session cookies...")

    async with async_playwright() as p:
        browser, context, page = await get_authenticated_context(p)

        # Navigate a few LinkedIn pages to refresh cookie timestamps
        try:
            await page.goto(LINKEDIN_FEED, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(2)
            await page.goto(
                "https://www.linkedin.com/jobs/",
                wait_until="domcontentloaded",
                timeout=15000
            )
            await asyncio.sleep(2)

            # Save refreshed cookies
            cookies = await context.cookies()
            meta = get_session_info()
            save_cookies(cookies, meta.get("email", "unknown"))
            print("  ✅ Cookies refreshed!")

        except Exception as e:
            print(f"  ⚠️  Refresh failed: {e}")
        finally:
            await browser.close()

    return True


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

async def main():
    if "--login" in sys.argv:
        await create_linkedin_session()

    elif "--check" in sys.argv:
        print("\n📋 LinkedIn Session Status")
        print("=" * 40)
        info = get_session_info()
        if info.get("status") == "no_session":
            print("  ❌ No session found")
            print("  Run: python automation/linkedin_session.py --login")
        else:
            print(f"  Email:      {info.get('email')}")
            print(f"  Saved at:   {info.get('saved_at', 'N/A')[:19]}")
            print(f"  Expires at: {info.get('expires_at', 'N/A')[:19]}")
            print(f"  Cookies:    {info.get('cookie_count', 0)}")
            valid = is_session_valid()
            print(f"  Status:     {'✅ Valid' if valid else '❌ Expired'}")

    elif "--clear" in sys.argv:
        clear_session()

    elif "--refresh" in sys.argv:
        await refresh_session_if_needed()

    else:
        print("\nLinkedIn Session Manager")
        print("Usage:")
        print("  python automation/linkedin_session.py --login    # Create new session")
        print("  python automation/linkedin_session.py --check    # Check session status")
        print("  python automation/linkedin_session.py --refresh  # Refresh cookies")
        print("  python automation/linkedin_session.py --clear    # Delete session")


if __name__ == "__main__":
    asyncio.run(main())
