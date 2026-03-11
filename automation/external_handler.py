import asyncio
import random
from pathlib import Path
from playwright.async_api import Page, BrowserContext
import json

async def fill_form_heuristically(page: Page, job: dict, resume_data: dict):
    """
    Try to fill a job application form using common selectors.
    Works best for Greenhouse, Lever, and standard HR platforms.
    """
    print(f"  🧠 Attempting to fill external form for {job.get('company')}...")
    
    # Common fields mapping (Selector heuristics -> Resume data keys)
    field_map = [
        (["first_name", "first-name", "given-name"], resume_data.get("name", "").split()[0]),
        (["last_name", "last-name", "family-name"], resume_data.get("name", "").split()[-1]),
        (["full_name", "full-name", "name"], resume_data.get("name", "")),
        (["email"], resume_data.get("email", "")),
        (["phone", "tel", "contact_number"], resume_data.get("phone", "")),
        (["linkedin"], resume_data.get("linkedin", "")),
        (["github"], resume_data.get("github", "")),
        (["portfolio", "website", "online-profile"], resume_data.get("portfolio", "")),
        (["location", "address", "city"], resume_data.get("location", "")),
    ]

    # 1. Fill Text Inputs
    for selectors, value in field_map:
        if not value: continue
        for sel in selectors:
            try:
                # Try name, id, and aria-label
                input_el = await page.query_selector(
                    f"input[name*='{sel}' i], input[id*='{sel}' i], "
                    f"input[aria-label*='{sel}' i], input[placeholder*='{sel}' i]"
                )
                if input_el and await input_el.is_visible():
                    current_val = await input_el.input_value()
                    if not current_val:
                        await input_el.fill(value)
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        break # Move to next field type
            except:
                continue

    # 2. Handle Dropdowns (Country/India logic)
    try:
        selects = await page.query_selector_all("select")
        for sel in selects:
            html = await sel.inner_html()
            if "India" in html:
                # Check if it's a country/location dropdown
                name_attr = (await sel.get_attribute("name") or "").lower()
                id_attr = (await sel.get_attribute("id") or "").lower()
                if any(k in name_attr or k in id_attr for k in ["country", "location", "region"]):
                    # Try to select India
                    await sel.select_option(label="India")
                    await asyncio.sleep(0.3)
    except:
        pass

    # 3. Upload Resume PDF
    resume_pdf = job.get("resume_pdf")
    if resume_pdf and Path(resume_pdf).exists():
        print(f"  ⬆️  Uploading tailored resume: {Path(resume_pdf).name}")
        try:
            # Look for file inputs
            file_input = await page.query_selector("input[type='file'][name*='resume' i], input[type='file'][id*='resume' i], input[type='file']")
            if file_input:
                await file_input.set_input_files(resume_pdf)
                await asyncio.sleep(2)
        except Exception as e:
            print(f"  ⚠️  Failed to upload resume: {e}")

    # 3. Handle Cover Letter Textarea
    cover_letter = job.get("cover_letter")
    if cover_letter:
        try:
            ta = await page.query_selector("textarea[name*='cover' i], textarea[id*='cover' i], textarea[aria-label*='cover' i]")
            if ta:
                current = await ta.input_value()
                if not current:
                    await ta.fill(cover_letter)
                    await asyncio.sleep(0.5)
        except:
            pass

    # 4. Handle "I agree" / Checkboxes
    try:
        checkboxes = await page.query_selector_all("input[type='checkbox']")
        for cb in checkboxes:
            # Try to see if it's a "consent" or "agree" checkbox
            parent_text = await page.evaluate("(el) => el.parentElement.innerText", cb)
            if any(term in parent_text.lower() for term in ["agree", "consent", "acknowledge", "understand"]):
                if not await cb.is_checked():
                    await cb.check()
                    await asyncio.sleep(0.3)
    except:
        pass

    print("  ✅ Form filled (best effort). User should review and submit if not autonomous.")

async def find_and_click_apply_button(page: Page):
    """Try to find the initial 'Apply' button on a company job page."""
    selectors = [
        "a:has-text('Apply Now')", "button:has-text('Apply Now')",
        "a:has-text('Apply for this job')", "button:has-text('Apply for this job')",
        "#apply_button", ".apply-button", "[data-test='apply-button']"
    ]
    for sel in selectors:
        try:
            btn = await page.query_selector(sel)
            if btn and await btn.is_visible():
                await btn.click()
                await asyncio.sleep(2)
                return True
        except:
            continue
    return False

async def handle_account_creation(page: Page, resume_data: dict):
    """Detect if we are on a login page and try to create an account."""
    print("  🔐 Checking if account creation is required...")
    
    # Check for Login/Sign In indicators
    login_indicators = ["login", "sign-in", "signin", "log-in"]
    if any(ind in page.url.lower() for ind in login_indicators) or await page.query_selector("input[type='password']"):
        print("  🔑 Login/Register page detected. Looking for 'Create Account'...")
        
        # Try to find Register/Sign Up/Create Account button
        register_btn = await page.query_selector(
            "a:has-text('Register'), a:has-text('Sign Up'), "
            "button:has-text('Create Account'), a:has-text('Create an account')"
        )
        
        if register_btn:
            print("  🖱️  Clicking register button...")
            await register_btn.click()
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            # Fill Registration Form
            print("  📝 Filling registration form...")
            await fill_form_heuristically(page, {}, resume_data)
            
            # Special handling for Password fields in registration
            passwords = await page.query_selector_all("input[type='password']")
            for pw in passwords:
                await pw.fill("Mustakim@JobAgent2026") # Use a standard strong placeholder
                await asyncio.sleep(0.3)
            
            # Look for "Register" or "Submit" button for the account
            submit_reg = await page.query_selector(
                "button:has-text('Register'), button:has-text('Sign Up'), "
                "button[type='submit']"
            )
            if submit_reg:
                print("  🔘 Submitting registration...")
                await submit_reg.click()
                await page.wait_for_load_state("networkidle", timeout=30000)
                return True
    return False

async def handle_external_site(context: BrowserContext, linkedin_page: Page, job: dict, resume_data: dict) -> bool:
    """
    Handles redirection from LinkedIn to an external company site and attempts to apply.
    """
    print(f"  🚀 Navigating to external site for {job.get('company')}...")
    
    # 1. Click the external Apply button on LinkedIn
    # This usually opens a new tab
    async with context.expect_page() as new_page_info:
        # Re-find the button just in case
        apply_btn = await linkedin_page.query_selector(
            ".jobs-apply-button--top-card button, button.jobs-apply-button"
        )
        if not apply_btn:
            print("  ❌ External Apply button disappeared.")
            return False
            
        await apply_btn.click()
        
    external_page = await new_page_info.value
    await external_page.bring_to_front()
    await external_page.wait_for_load_state("networkidle", timeout=60000)
    
    print(f"  🌐 External site loaded: {external_page.url}")
    
    # 2. Handle Account Creation if needed
    await handle_account_creation(external_page, resume_data)

    # 3. Check if we are on the form or need to click 'Apply' again
    if "apply" not in external_page.url.lower():
        await find_and_click_apply_button(external_page)
        await external_page.wait_for_load_state("networkidle", timeout=30000)

    # 4. Fill the form
    await fill_form_heuristically(external_page, job, resume_data)
    
    # 5. Attempt Submit (Highly Autonomous)
    is_common_platform = any(p in external_page.url.lower() for p in ["greenhouse.io", "lever.co", "workday", "breezy.hr", "tal.net"])
    
    submit_btn = await external_page.query_selector(
        "button[type='submit']:has-text('Submit'), "
        "button[type='submit']:has-text('Apply'), "
        "button[id*='submit' i], "
        "input[type='submit']"
    )
    
    if submit_btn and is_common_platform:
        print(f"  🔘 Submit button found on {job.get('company')} (platform detected). Clicking for autonomous submission in 3s...")
        await asyncio.sleep(3) # Brief wait for user to see
        await submit_btn.click() # FULLY AUTONOMOUS SUBMISSION
        print("  🎉 External application submitted autonomously!")
        return True 
    elif submit_btn:
        print("  🔘 Submit button found on custom site. Waiting 10s for user review before continuing...")
        await asyncio.sleep(10) # Give user a chance to check

    return True # Return true because we successfully filled as much as we could
