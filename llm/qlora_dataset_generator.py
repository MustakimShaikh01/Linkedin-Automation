"""
QLoRA Dataset Generator
Generates 2000 fine-tuning samples in minutes by combining your resume
with diverse job descriptions. Used to fine-tune Phi-3 Mini or Mistral 7B
specifically for your resume tailoring task.
"""

import json
import random
from pathlib import Path
from datetime import datetime

RESUME_PATH = Path(__file__).parent.parent / "resume" / "resume.json"
OUTPUT_DIR = Path(__file__).parent.parent / "llm" / "qlora_dataset"

# ─── Job Description Templates ──────────────────────────────────────────────────
JOB_TEMPLATES = {
    "AI Engineer": [
        "Build LLM-powered applications using Python and LangChain. RAG systems experience required. FastAPI, Docker, vector databases. Remote.",
        "Design and deploy retrieval-augmented generation pipelines for enterprise clients. Experience with Ollama, llama.cpp, and HuggingFace required.",
        "Develop AI agents with tool use and multi-step reasoning. LangChain, OpenAI API, FastAPI. 2+ years LLM experience.",
        "Build document intelligence systems processing thousands of records daily. Python, RAG, FAISS, PostgreSQL required.",
        "LLM Engineer to design prompts, fine-tune models, and deploy AI APIs. HuggingFace, QLoRA, PEFT knowledge essential.",
    ],
    "Machine Learning Engineer": [
        "Build and deploy ML models for production. Python, scikit-learn, PyTorch, MLflow, Docker, AWS. REST API experience.",
        "Design NLP pipelines for text classification and semantic search. Transformers, FAISS, embeddings. FastAPI backend.",
        "Develop recommendation systems and build data pipelines. Python pandas, ML frameworks, PostgreSQL, Redis.",
        "ML Engineer to build analytics pipelines and dashboards. 30K+ records processing experience preferred.",
        "Build semantic search systems with vector databases. Embeddings, sentence transformers, ChromaDB or FAISS.",
    ],
    "Backend AI Developer": [
        "Python backend developer with AI/ML exposure. FastAPI, PostgreSQL, Docker, REST APIs, Redis. LLM API integration.",
        "Build scalable API services for AI applications. Python, FastAPI, async programming, SQLite, PostgreSQL. LangChain a plus.",
        "Develop ERP and financial workflow automation using Python. Backend APIs, database optimization, reporting systems.",
        "Backend engineer for data analytics platform. Python, pandas, SQL, dashboards, automation. Manufacturing domain preferred.",
        "Build AI services with sub-400ms latency requirements. FastAPI, caching, async Python, monitoring with MLflow.",
    ],
    "NLP Engineer": [
        "Design NLP pipelines for information extraction. Transformers, spaCy, HuggingFace, semantic search, embeddings.",
        "Build text summarization and question-answering systems. RAG, LangChain, vector databases, prompt engineering.",
        "NLP Engineer to work on document processing. HuggingFace fine-tuning, PEFT, QLoRA, Python. Remote position.",
        "Develop semantic search over large document corpora. BAAI embeddings, FAISS indexing, Python, FastAPI.",
        "Build knowledge base Q&A using RAG architecture. LangChain, OpenAI/Ollama, vector stores, citation retrieval.",
    ],
    "Data Engineer": [
        "Build data pipelines processing millions of records. Python, SQL, PostgreSQL, pandas, Apache Spark. AWS S3.",
        "Develop ETL processes and analytics dashboards. Python, data warehousing, reporting automation, SQL optimization.",
        "Data Engineer for manufacturing analytics. Python pipelines, 30K+ records daily, dashboard reporting, monitoring.",
        "Build and maintain data infrastructure. PostgreSQL, MongoDB, Redis, Python, Docker, Linux. REST APIs.",
        "Analytics engineer to automate reporting workflows. Python, SQL, dashboards, reducing manual effort by 40%+.",
    ],
}

# ─── Tailored Output Templates ───────────────────────────────────────────────────
# These are structured outputs the model should learn to generate

def generate_tailored_summary(job_role: str, focus_area: str) -> str:
    summaries = {
        "AI Engineer": f"Applied AI engineer specializing in {focus_area} with hands-on experience building production LLM systems. Proven track record of designing RAG pipelines, deploying FastAPI AI services handling 300+ daily queries, and implementing local LLM inference reducing costs by 30%. Passionate about building reliable, scalable AI solutions.",
        "Machine Learning Engineer": f"ML engineer with practical experience building production pipelines processing 30K+ records daily. Skilled in {focus_area}, Python, FastAPI, and data infrastructure. Experienced in automating workflows that reduce manual effort by 40% and developing systems that improve operational visibility.",
        "Backend AI Developer": f"Python backend developer with deep AI/LLM integration experience. Built document intelligence systems processing 5,000+ documents monthly and designed FastAPI services with sub-400ms latency. Expert in {focus_area} and scalable API architecture.",
        "NLP Engineer": f"NLP engineer with expertise in {focus_area}, RAG architectures, and semantic search systems. Built knowledge retrieval pipelines using LangChain and vector databases. Strong foundation in HuggingFace Transformers, embeddings, and prompt engineering for production AI systems.",
        "Data Engineer": f"Data engineer experienced in {focus_area} and building analytics pipelines at scale. Developed systems processing 30,000+ production records daily, automated reporting workflows reducing effort by 40%, and built dashboards improving operational decision-making.",
    }
    return summaries.get(job_role, summaries["AI Engineer"])


def generate_key_bullets(job_role: str) -> list[str]:
    all_bullets = {
        "AI Engineer": [
            "Built LLM-based document intelligence system processing 5,000+ documents monthly, reducing manual analysis by 50%",
            "Developed RAG pipelines using LangChain and FAISS for knowledge retrieval across internal datasets",
            "Designed FastAPI AI services handling 300+ daily LLM queries with response latency under 400ms",
            "Implemented local LLM inference using Ollama and llama.cpp, reducing external API costs by 30%",
            "Built autonomous research agent that gathers multi-source information and generates structured reports",
        ],
        "Machine Learning Engineer": [
            "Built analytics pipelines processing 30,000+ production records daily for manufacturing performance monitoring",
            "Designed semantic search systems using FAISS, sentence transformers, and vector databases",
            "Developed ML-powered dashboards improving operational visibility and supporting data-driven decisions",
            "Automated reporting workflows reducing manual effort by approximately 40%",
            "Implemented NLP pipelines for information extraction using HuggingFace Transformers",
        ],
        "Backend AI Developer": [
            "Designed FastAPI-based AI services handling 300+ daily LLM queries with sub-400ms response latency",
            "Built backend APIs and automation modules for ERP and financial workflow systems",
            "Implemented local LLM inference using Ollama reducing external API usage costs by 30%",
            "Developed REST APIs and database integrations across PostgreSQL, MongoDB, Redis, and SQLite",
            "Built ETL pipelines processing 30,000+ records daily with automated monitoring and alerting",
        ],
        "NLP Engineer": [
            "Developed RAG system using LangChain and vector databases for semantic search across documentation",
            "Built document intelligence pipeline processing 5,000+ monthly documents with NLP extraction",
            "Implemented semantic search using BAAI/bge-small embeddings and FAISS vector indexing",
            "Designed prompt engineering frameworks reducing hallucination in production LLM outputs",
            "Built autonomous research agent using LLM APIs for multi-source information synthesis",
        ],
        "Data Engineer": [
            "Built analytics pipelines processing 30,000+ manufacturing records daily with real-time monitoring",
            "Automated reporting workflows reducing manual effort by 40% using Python and SQL optimization",
            "Developed dashboards improving operational visibility and supporting data-driven decision making",
            "Built backend data APIs across PostgreSQL, MongoDB, Redis for diverse analytics workloads",
            "Implemented ETL processes and data infrastructure using Docker and AWS (EC2, S3)",
        ],
    }
    bullets = all_bullets.get(job_role, all_bullets["AI Engineer"])
    return random.sample(bullets, min(4, len(bullets)))


def generate_relevant_skills(job_role: str) -> str:
    skills_map = {
        "AI Engineer": "Python, LangChain, RAG, Ollama, llama.cpp, FastAPI, FAISS, HuggingFace, Prompt Engineering, Docker, PostgreSQL, Vector Databases",
        "Machine Learning Engineer": "Python, scikit-learn, PyTorch, HuggingFace Transformers, FAISS, Embeddings, FastAPI, MLflow, PostgreSQL, Docker, AWS",
        "Backend AI Developer": "Python, FastAPI, Flask, REST APIs, PostgreSQL, MongoDB, Redis, SQLite, Docker, Git, AWS, LangChain, Ollama",
        "NLP Engineer": "Python, HuggingFace, LangChain, RAG, FAISS, Sentence Transformers, Embeddings, Semantic Search, FastAPI, Prompt Engineering",
        "Data Engineer": "Python, pandas, SQL, PostgreSQL, MongoDB, Redis, Docker, AWS (EC2, S3), FastAPI, MLflow, Analytics Pipelines",
    }
    return skills_map.get(job_role, skills_map["AI Engineer"])


# ─── Dataset Generator ───────────────────────────────────────────────────────────

def generate_sample(job_role: str, job_description: str, index: int) -> dict:
    """Generate a single instruction-following training sample."""

    # Load real resume for context
    with open(RESUME_PATH) as f:
        resume = json.load(f)

    resume_summary = resume.get("summary", "")
    resume_name = resume.get("name", "Mustakim Shaikh")

    focus_areas = [
        "LLM systems and RAG", "document intelligence", "semantic search",
        "AI automation", "NLP pipelines", "local LLM deployment",
        "vector databases", "prompt engineering", "FastAPI AI services",
    ]
    focus = random.choice(focus_areas)

    tailored_summary = generate_tailored_summary(job_role, focus)
    key_bullets = generate_key_bullets(job_role)
    relevant_skills = generate_relevant_skills(job_role)

    bullets_text = "\n".join(f"• {b}" for b in key_bullets)

    output = f"""TAILORED SUMMARY:
{tailored_summary}

KEY ACHIEVEMENTS (top 4 most relevant):
{bullets_text}

RELEVANT SKILLS (for this job):
{relevant_skills}"""

    return {
        "instruction": (
            f"You are an ATS resume optimizer. "
            f"Rewrite the resume to match the job description using ONLY real information from the resume. "
            f"Do NOT invent experience or skills."
        ),
        "input": (
            f"Resume Owner: {resume_name}\n"
            f"Original Summary: {resume_summary}\n\n"
            f"Job Role: {job_role}\n"
            f"Job Description: {job_description}"
        ),
        "output": output,
    }


def generate_cover_letter_sample(job_role: str, job_description: str,
                                   company: str) -> dict:
    """Generate a cover letter training sample."""
    with open(RESUME_PATH) as f:
        resume = json.load(f)

    name = resume.get("name", "Mustakim Shaikh")

    cover_letters = {
        "AI Engineer": f"""I am excited to apply for the {job_role} position at {company}. With hands-on experience building RAG systems, LLM-powered APIs, and document intelligence pipelines in production environments, I am confident I can make an immediate contribution.

In my current role at Vikash Tech Solution, I built an LLM-based system processing 5,000+ documents monthly and designed FastAPI services handling 300+ daily AI queries with sub-400ms latency. My experience with Ollama, LangChain, and local LLM deployment directly aligns with your requirements.

I would welcome the opportunity to discuss how my background in AI engineering can help {company} achieve its goals. Thank you for considering my application.

Best regards,
{name}""",
        "Machine Learning Engineer": f"""I am writing to express my strong interest in the {job_role} role at {company}. My background in building production ML pipelines and analytics systems makes me an ideal candidate for this position.

At Siddharth Carbon Chemicals, I built analytics pipelines processing 30,000+ records daily and automated reporting workflows that reduced manual effort by 40%. My technical expertise in Python, ML frameworks, and data infrastructure aligns perfectly with your team's needs.

I look forward to the opportunity to bring my data engineering and ML experience to {company}. Thank you for your consideration.

Best regards,
{name}""",
    }

    output = cover_letters.get(job_role, cover_letters["AI Engineer"])

    return {
        "instruction": (
            "Write a professional 3-paragraph cover letter using ONLY real information "
            "from the resume. Do not invent achievements or experiences."
        ),
        "input": (
            f"Applicant: {name}\n"
            f"Target Role: {job_role}\n"
            f"Company: {company}\n"
            f"Job Description: {job_description[:500]}"
        ),
        "output": output,
    }


def generate_qlora_dataset(num_samples: int = 2000) -> list[dict]:
    """Generate a complete QLoRA fine-tuning dataset."""
    print(f"🎯 Generating {num_samples} QLoRA training samples...")

    dataset = []
    companies = [
        "TechStartup", "AI Labs", "DataCorp", "NeuralWorks", "CloudAI",
        "SmartSystems", "DeepTech", "VectorAI", "LLMCo", "AgileTech",
        "InnovatAI", "DataStream", "AIFoundry", "PythonLabs", "ByteAI",
    ]

    roles = list(JOB_TEMPLATES.keys())
    samples_per_role = num_samples // len(roles)

    for role in roles:
        job_descs = JOB_TEMPLATES[role]

        for i in range(samples_per_role):
            # Cycle through job descriptions with variation
            base_desc = job_descs[i % len(job_descs)]

            # Add variation to avoid exact duplicates
            prefixes = [
                "We are looking for a", "Join our team as a",
                "Exciting opportunity for a", "Hiring a senior",
                "Remote position for a", "We need an experienced",
            ]
            company = random.choice(companies)
            prefix = random.choice(prefixes)
            varied_desc = f"{prefix} {role}. {base_desc}. Company: {company}."

            # Generate resume tailoring sample
            sample = generate_sample(role, varied_desc, i)
            dataset.append(sample)

            # Generate cover letter sample (30% of samples)
            if random.random() < 0.3:
                cl_sample = generate_cover_letter_sample(role, varied_desc, company)
                dataset.append(cl_sample)

        print(f"  ✅ Generated samples for: {role}")

    # Shuffle dataset
    random.shuffle(dataset)

    # Trim to exact count
    dataset = dataset[:num_samples]
    print(f"\n✅ Total samples generated: {len(dataset)}")
    return dataset


def save_dataset(dataset: list[dict]):
    """Save dataset in multiple formats for different training frameworks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. JSONL format (for LLaMA-Factory, Axolotl)
    jsonl_path = OUTPUT_DIR / f"qlora_dataset_{timestamp}.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")
    print(f"  💾 Saved JSONL: {jsonl_path.name}")

    # 2. JSON format (for HuggingFace datasets)
    json_path = OUTPUT_DIR / f"qlora_dataset_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  💾 Saved JSON: {json_path.name}")

    # 3. Train/val split (90/10)
    split = int(len(dataset) * 0.9)
    train_data = dataset[:split]
    val_data = dataset[split:]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    with open(train_path, "w") as f:
        for s in train_data:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_data:
            f.write(json.dumps(s) + "\n")

    print(f"  💾 Saved train ({len(train_data)}) and val ({len(val_data)}) splits")

    return jsonl_path


if __name__ == "__main__":
    print("🧬 QLoRA Dataset Generator")
    print("=" * 50)

    dataset = generate_qlora_dataset(num_samples=2000)
    save_dataset(dataset)

    # Print sample
    print("\n📋 Sample training example:")
    print("-" * 40)
    sample = dataset[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Input: {sample['input'][:200]}...")
    print(f"Output: {sample['output'][:300]}...")
