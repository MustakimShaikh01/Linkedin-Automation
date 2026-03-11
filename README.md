# 🤖 AI Job Agent — Mac M2 Optimized

> A fully autonomous AI job hunting pipeline built for Apple Silicon (M2/M3).  
> Scrapes jobs → Matches with your resume → Tailors resume with LLM → Auto-applies.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2B%20Mistral%207B-black?style=flat)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-blue?style=flat)
![Playwright](https://img.shields.io/badge/Scraper-Playwright-45ba4b?style=flat)
![FastAPI](https://img.shields.io/badge/Dashboard-FastAPI-009688?style=flat)

---

## ✨ Features

| Feature | Tech | Mac M2 Friendly |
|---|---|---|
| Job Scraping | Playwright headless | ✅ Low CPU |
| Resume Matching | FAISS + BGE embeddings | ✅ No LLM needed |
| Resume Tailoring | Mistral 7B / Phi-3 via Ollama | ✅ Only for matches |
| Cover Letter | Mistral 7B (low-hallucination prompt) | ✅ |
| Auto Apply | Playwright + LinkedIn Easy Apply | ✅ 5/day limit |
| Dashboard | FastAPI + real-time stats | ✅ localhost:8000 |
| QLoRA Dataset | 2000 samples in 2 minutes | ✅ |
| Scheduler | Runs every 6 hours | ✅ CPU idle 95% |

---

## 🚀 Quick Start

```bash
# 1. Clone / navigate to project
cd Automation-end-to-end

# 2. One-click setup
chmod +x setup.sh && ./setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Update YOUR resume
# Edit: resume/resume.json  (already pre-filled with your data)

# 5. Start dashboard
python dashboard.py
# → Open http://localhost:8000

# 6. Run pipeline
python main.py --once     # One run (test)
python main.py            # Scheduled (every 6 hours)
```

---

## 📁 Project Structure

```
Automation-end-to-end/
│
├── 📄 resume/
│   ├── resume.json              ← Your structured resume (edit this!)
│   └── tailored/                ← LLM-generated tailored resumes
│
├── 🔍 scraper/
│   └── job_scraper.py           ← Playwright job scraper (LinkedIn etc.)
│
├── 🎯 matcher/
│   └── similarity_engine.py     ← FAISS + BGE semantic matching
│
├── 🤖 llm/
│   ├── resume_optimizer.py      ← Mistral 7B resume tailor + cover letter
│   └── qlora_dataset_generator.py ← Generate 2000 QLoRA samples
│
├── 🚗 automation/
│   └── auto_apply.py            ← Playwright auto-apply bot (5/day)
│
├── 📊 database/
│   ├── applications.db          ← SQLite (all jobs + applications)
│   ├── scraped_jobs.json        ← Job cache
│   ├── apply_log.json           ← Application history
│   └── manual_review/           ← Jobs needing manual application
│
├── 🧠 embeddings/
│   ├── embeddings_cache.json    ← Cached embeddings (avoids recompute)
│   └── resume.faiss             ← FAISS index of your resume
│
├── 📡 dashboard.py              ← FastAPI monitoring dashboard
├── 🔄 main.py                   ← Pipeline orchestrator + scheduler
├── 📦 requirements.txt
└── 🛠️ setup.sh                  ← One-click setup
```

---

## ⚙️ How It Works

```
Every 6 hours:

1. 📡 SCRAPE      Playwright headless → LinkedIn jobs (headless, low CPU)
       ↓
2. 🎯 MATCH       FAISS + BGE-small → cosine similarity vs resume
       ↓
3. ✅ FILTER      Score ≥ 60%? → Pass to LLM  |  Score < 60%? → Skip
       ↓
4. ✍️ TAILOR      Mistral 7B via Ollama → tailored resume + cover letter
       ↓
5. 🤖 APPLY       LinkedIn Easy Apply → max 5/day (safety limit)
       ↓
6. 📊 REPORT      Dashboard updates with stats, CPU, RAM
```

> **Key optimization**: FAISS filters 90% of jobs _before_ any LLM calls → saves enormous CPU

---

## 🧠 Models Used

| Model | Use Case | RAM |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | Embeddings / similarity | ~200MB |
| `mistral` (7B) | Resume tailoring, cover letters | ~4-5GB |
| `phi3` (3.8B) | Fast fallback if Mistral slow | ~2-3GB |

```bash
# Install models
ollama pull mistral    # Primary (recommended)
ollama pull phi3       # Fast fallback
```

---

## 📊 Dashboard

```bash
python dashboard.py
```

Open [http://localhost:8000](http://localhost:8000) to see:

- 📬 Applications sent today (with daily cap)
- 💻 CPU & RAM usage (live)
- 🤖 Ollama status + models
- 📋 All scraped jobs with match scores
- 🔄 Manual pipeline trigger button

---

## 🧬 Fine-tune Your Own Model

Generate 2000 training samples from your resume:

```bash
python llm/qlora_dataset_generator.py
```

Output:
- `llm/qlora_dataset/train.jsonl` (1800 samples)
- `llm/qlora_dataset/val.jsonl` (200 samples)

Then fine-tune with [MLX-LM](https://github.com/ml-explore/mlx-examples) (native Apple Silicon):

```bash
pip install mlx-lm
mlx_lm.lora --model mistralai/Mistral-7B-Instruct-v0.2 \
             --train --data llm/qlora_dataset/ \
             --iters 1000
```

---

## 🔒 Safety Features

| Safety | Setting |
|---|---|
| Max applications/day | `5` (configurable) |
| CPU guard | Pauses if CPU > 60% |
| Random delays | 10–20s between applies |
| Similarity threshold | `60%` minimum match |
| Anti-hallucination | Strict prompt engineering |
| Manual review | Complex portals saved for review |

---

## 📝 Customizing

### Update your resume
Edit `resume/resume.json` — all real data is here.  
Then rebuild the FAISS index:
```bash
python -c "from matcher.similarity_engine import build_resume_faiss_index; build_resume_faiss_index()"
```

### Change target jobs
Edit `scraper/job_scraper.py`:
```python
TARGET_KEYWORDS = ["AI Engineer", "ML Engineer", "LLM Engineer", ...]
TARGET_LOCATIONS = ["Remote", "India", ...]
```

### Adjust thresholds
Edit `matcher/similarity_engine.py`:
```python
SIMILARITY_THRESHOLD = 0.60  # Increase for stricter matching
```

---

## 📦 Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- [Ollama](https://ollama.ai) installed
- 8GB RAM minimum (16GB recommended)

---

Built by **Mustakim Shaikh** — AI/ML Engineer  
[GitHub](https://github.com/MustakimShaikh01) • [LinkedIn](https://linkedin.com/in/mustakim-sh) • [Portfolio](https://mustakim-portfolio-jet.vercel.app)
# Linkedin-Automation
