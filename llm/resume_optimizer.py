"""
Resume Optimizer — Mac M2 Optimized (Ollama + Mistral 7B)
LLM is ONLY called for jobs that pass the similarity threshold.
Low-hallucination prompt engineering enforces using only real resume data.
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime

RESUME_PATH = Path(__file__).parent.parent / "resume" / "resume.json"
OUTPUT_DIR = Path(__file__).parent.parent / "resume" / "tailored"

OLLAMA_BASE_URL = "http://localhost:11434"
PRIMARY_MODEL = "mistral"       # Mistral 7B — best quality
FALLBACK_MODEL = "phi3"         # Phi-3 Mini — if Mistral is too slow
OLLAMA_TIMEOUT = 120            # seconds (generous for M2)
MAX_TOKENS = 2048               # Keep response concise


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_available_model() -> str:
    """Return the best available Ollama model."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]

        if any(PRIMARY_MODEL in m for m in models):
            return PRIMARY_MODEL
        if any(FALLBACK_MODEL in m for m in models):
            return FALLBACK_MODEL

        # Return first available model
        return models[0] if models else PRIMARY_MODEL

    except Exception:
        return PRIMARY_MODEL


def load_resume() -> dict:
    """Load the structured resume JSON."""
    with open(RESUME_PATH) as f:
        return json.load(f)


def build_resume_context(resume: dict) -> str:
    """
    Build a concise resume context string for the LLM prompt.
    Only includes real facts — no fabrication possible.
    """
    lines = []

    lines.append(f"Name: {resume['name']}")
    lines.append(f"Summary: {resume['summary']}")
    lines.append("")

    # Skills (flat list)
    all_skills = []
    skills = resume.get("skills", {})
    if isinstance(skills, dict):
        for cat, skill_list in skills.items():
            all_skills.extend(skill_list)
    elif isinstance(skills, list):
        all_skills = skills
    lines.append(f"Skills: {', '.join(all_skills)}")
    lines.append("")

    # Experience
    lines.append("Experience:")
    for exp in resume.get("experience", []):
        lines.append(
            f"- {exp['title']} at {exp['company']} "
            f"({exp.get('start', '')} – {exp.get('end', '')})"
        )
        for bullet in exp.get("bullets", []):
            lines.append(f"  • {bullet}")
    lines.append("")

    # Projects
    lines.append("Projects:")
    for proj in resume.get("projects", []):
        lines.append(f"- {proj['name']}: {proj['description']}")
        lines.append(f"  Tech: {', '.join(proj.get('tech', []))}")
    lines.append("")

    # Education
    for edu in resume.get("education", []):
        lines.append(
            f"Education: {edu['degree']}, {edu['institution']} "
            f"({edu.get('start', '')}–{edu.get('end', '')}), "
            f"CGPA: {edu.get('cgpa', 'N/A')}"
        )

    return "\n".join(lines)


def build_resume_tailor_prompt(resume_context: str,
                                job_title: str,
                                job_description: str,
                                top_resume_matches: list[str]) -> str:
    """
    Prompt optimized for LaTeX injection with strong instructions for 2-3 yrs Backend/GenAI experience.
    """
    return f"""You are an expert ATS resume optimizer.

CRITICAL RULES:
1. Emphasize that the candidate has 2 to 3 years of experience in AI, Backend engineering, Python, Flask, and Generative AI.
2. Align the summary and bullet points directly with the Job Description.
3. Keep the content professional and ATS-friendly.
4. Output EXACTLY valid JSON with these 4 keys (no markdown blocks, no other text):
"TAILORED_SUMMARY": "A 2-3 sentence summary...",
"TAILORED_SKILLS": "Languages: Python... Frameworks: Flask, FastAPI... AI: GenAI, RAG, LLMOps...",
"TAILORED_EXP_1": "One powerful bullet point emphasizing GenAI/RAG experience matching the job",
"TAILORED_EXP_2": "One powerful bullet point emphasizing Backend/Flask Python experience matching the job"

=== JOB TITLE ===
{job_title}

=== JOB DESCRIPTION ===
{job_description[:1500]}

Remember, ONLY output valid JSON.
"""


def build_cover_letter_prompt(resume_context: str,
                               job_title: str,
                               company: str,
                               job_description: str) -> str:
    """Prompt for generating a professional cover letter."""
    return f"""You are a professional cover letter writer.

CRITICAL RULES:
1. Use ONLY real information from the resume below.
2. Do NOT make up achievements, companies, or skills.
3. Be concise — 3 paragraphs maximum.
4. Sound professional, enthusiastic, and specific.

=== RESUME CONTEXT ===
{resume_context}

=== TARGET JOB ===
Title: {job_title}
Company: {company}
Description: {job_description[:800]}

Write a compelling 3-paragraph cover letter:
- Paragraph 1: Opening — why you're excited about this role
- Paragraph 2: Most relevant experience (from resume only)
- Paragraph 3: Closing and call to action

Do not use placeholders like [Your Name] — the name is already in the resume.
"""


def call_ollama(prompt: str, model: str = None) -> str:
    """
    Call Ollama LLM API with the given prompt.
    Non-streaming for simplicity and reliability.
    """
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Please start it with: ollama serve"
        )

    if model is None:
        model = get_available_model()

    print(f"  🤖 Calling Ollama ({model})...")
    start = time.time()

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": 0.3,    # Low temperature = less hallucination
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            },
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start

        print(f"  ✅ LLM response in {elapsed:.1f}s")
        return result.get("response", "").strip()

    except requests.Timeout:
        raise RuntimeError(f"Ollama timed out after {OLLAMA_TIMEOUT}s. Try phi3 model.")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


def tailor_resume_for_job(job: dict) -> dict:
    """
    Main function: Tailor resume for a specific job.

    Expects job dict with:
        - title, company, description, top_resume_matches

    Returns enhanced job dict with:
        - tailored_resume, cover_letter
    """
    print(f"\n✍️  Tailoring resume for: {job.get('title')} @ {job.get('company')}")

    resume = load_resume()
    resume_context = build_resume_context(resume)
    model = get_available_model()

    # --- Step 1: Tailor Resume ---
    tailor_prompt = build_resume_tailor_prompt(
        resume_context=resume_context,
        job_title=job.get("title", ""),
        job_description=job.get("description", ""),
        top_resume_matches=job.get("top_resume_matches", []),
    )

    tailored_resume = call_ollama(tailor_prompt, model=model)

    import re
    import subprocess
    
    # Extract JSON from LLM output (in case it includes markdown backticks)
    json_match = re.search(r"\{.*\}", tailored_resume, re.DOTALL)
    parsed = {"TAILORED_SUMMARY": tailored_resume, "TAILORED_SKILLS": "", "TAILORED_EXP_1": "", "TAILORED_EXP_2": ""}
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except Exception as e:
            print(f"  ⚠️  Failed to parse JSON from LLM: {e}")
            
    # Clean latex special chars function
    def escape_tex(text):
        if not isinstance(text, str): return ""
        # Just basic escaping for safety
        return text.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")

    # --- Step 2: Generate Cover Letter ---
    cover_prompt = build_cover_letter_prompt(
        resume_context=resume_context,
        job_title=job.get("title", ""),
        company=job.get("company", ""),
        job_description=job.get("description", ""),
    )

    cover_letter = call_ollama(cover_prompt, model=model)

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = (
        f"{job.get('company', 'unknown')}_{job.get('title', 'unknown')}"
        .replace(" ", "_").replace("/", "-")[:50]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ── Compile the LaTeX Template ──
    template_path = Path(__file__).parent.parent / "resume" / "resume_template.tex"
    if template_path.exists():
        tex_content = template_path.read_text()
        tex_content = tex_content.replace("%TAILORED_SUMMARY%", escape_tex(parsed.get("TAILORED_SUMMARY", "")))
        tex_content = tex_content.replace("%TAILORED_SKILLS%", escape_tex(parsed.get("TAILORED_SKILLS", "")))
        tex_content = tex_content.replace("%TAILORED_EXP_1%", escape_tex(parsed.get("TAILORED_EXP_1", "")))
        tex_content = tex_content.replace("%TAILORED_EXP_2%", escape_tex(parsed.get("TAILORED_EXP_2", "")))
        
        tex_file = OUTPUT_DIR / f"{safe_name}.tex"
        tex_file.write_text(tex_content)
        
        print("  📄 Compiling tailored resume PDF...")
        try:
            subprocess.run(
                ["tectonic", str(tex_file)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
            )
        except Exception as e:
            print(f"  ⚠️  LaTeX compilation failed (is tectonic installed?): {e}")
    else:
        print("  ⚠️  resume_template.tex not found!")

    output_file = OUTPUT_DIR / f"{timestamp}_{safe_name}.json"
    pdf_path = OUTPUT_DIR / f"{safe_name}.pdf"

    output = {
        "job_title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "url": job.get("url"),
        "similarity_score": job.get("similarity_score"),
        "tailored_resume": json.dumps(parsed),
        "cover_letter": cover_letter,
        "resume_pdf": str(pdf_path) if pdf_path.exists() else "",
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  💾 Saved to: {output_file.name}")

    job["tailored_resume"] = json.dumps(parsed)
    job["cover_letter"] = cover_letter
    job["resume_pdf"] = str(pdf_path) if pdf_path.exists() else ""

    return job


def batch_tailor_top_jobs(jobs: list[dict], max_jobs: int = 5) -> list[dict]:
    """
    Tailor resumes for the top N jobs (sorted by similarity score).
    Limits LLM calls to keep Mac M2 CPU manageable.
    """
    # Sort by score descending
    sorted_jobs = sorted(jobs, key=lambda j: j.get("similarity_score", 0), reverse=True)
    top_jobs = sorted_jobs[:max_jobs]

    print(f"\n🎯 Tailoring resumes for top {len(top_jobs)} jobs (of {len(jobs)} matched)")

    results = []
    for i, job in enumerate(top_jobs):
        print(f"\n[{i+1}/{len(top_jobs)}] Processing...")
        try:
            tailored = tailor_resume_for_job(job)
            results.append(tailored)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append(job)

        # Cool-down between LLM calls to avoid overheating M2
        if i < len(top_jobs) - 1:
            print("  😴 Cooling down 15s before next LLM call...")
            time.sleep(15)

    return results


if __name__ == "__main__":
    # Test with a sample job
    print("🧪 Testing Resume Optimizer")
    print("=" * 50)

    sample_job = {
        "title": "AI Engineer",
        "company": "TechStartup",
        "location": "Remote",
        "url": "https://example.com/job/123",
        "similarity_score": 0.78,
        "description": """
        We need an AI Engineer experienced in RAG, LangChain, and FastAPI.
        You will build LLM-powered document processing systems.
        Experience with Ollama, local LLMs, and vector databases required.
        Python, Docker, and REST API knowledge essential.
        """,
        "top_resume_matches": [
            "Built RAG systems using LangChain and vector databases",
            "Designed FastAPI-based AI services with 300+ daily queries",
            "Implemented local LLM inference using Ollama and llama.cpp",
        ]
    }

    result = tailor_resume_for_job(sample_job)
    print("\n=== TAILORED RESUME ===")
    print(result["tailored_resume"])
    print("\n=== COVER LETTER ===")
    print(result["cover_letter"])
