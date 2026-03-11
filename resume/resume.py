import json
from pathlib import Path

def load_resume_data() -> dict:
    """Load the master resume data from resume.json."""
    resume_path = Path(__file__).parent / "resume.json"
    if not resume_path.exists():
        return {}
    with open(resume_path, "r") as f:
        return json.load(f)
