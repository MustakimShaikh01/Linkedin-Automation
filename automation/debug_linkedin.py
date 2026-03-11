import asyncio
from playwright.async_api import async_playwright
import json

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        with open("database/sessions/linkedin_cookies.json") as f:
            cookies = json.load(f)
        await context.add_cookies(cookies)
        
        page = await context.new_page()
        await page.goto("https://www.linkedin.com/feed", wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)
        print(f"URL: {page.url}")
        print(f"TITLE: {await page.title()}")
        html = await page.content()
        print(f"HTML Preview: {html[:1000]}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
