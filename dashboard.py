"""
Dashboard — Real-time monitoring for the AI Job Agent pipeline.
FastAPI backend + simple HTML frontend served locally.
Check status, view applications, and trigger runs manually.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
import psutil
import subprocess
from datetime import datetime, date
from pathlib import Path
import uvicorn

DB_PATH = Path(__file__).parent / "database" / "applications.db"
LOG_FILE = Path(__file__).parent / "database" / "pipeline_log.json"
APPLY_LOG = Path(__file__).parent / "database" / "apply_log.json"

app = FastAPI(title="AI Job Agent Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Database Helpers ───────────────────────────────────────────────────────────

def get_db_stats() -> dict:
    if not DB_PATH.exists():
        return {"total_scraped": 0, "total_matched": 0, "total_applied": 0, "today_applied": 0}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM jobs")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM jobs WHERE similarity_score >= 0.6")
    matched = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM jobs WHERE applied = 1")
    applied = cursor.fetchone()[0]

    today = date.today().isoformat()
    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied = 1 AND date(applied_at) = ?",
        (today,)
    )
    today_applied = cursor.fetchone()[0]

    conn.close()
    return {
        "total_scraped": total,
        "total_matched": matched,
        "total_applied": applied,
        "today_applied": today_applied,
    }


def count_jobs(status: str = None, min_score: float = None, query: str = None, today_only: bool = False) -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = "SELECT COUNT(*) FROM jobs WHERE 1=1"
    params = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if min_score is not None:
        sql += " AND similarity_score >= ?"
        params.append(min_score)
    if query:
        sql += " AND (title LIKE ? OR company LIKE ?)"
        params.append(f"%{query}%")
        params.append(f"%{query}%")
    if today_only:
        sql += " AND date(scraped_at) = date('now')"
    cursor.execute(sql, params)
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_recent_jobs(limit: int = 10, offset: int = 0, status: str = None, min_score: float = None, query: str = None, sort_by: str = "scraped_at", today_only: bool = False) -> list:
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = """
        SELECT id, source, title, company, location, url, keyword,
               similarity_score, applied, applied_at, status, scraped_at
        FROM jobs WHERE 1=1
    """
    params = []

    if status:
        sql += " AND status = ?"
        params.append(status)
    
    if min_score is not None:
        sql += " AND similarity_score >= ?"
        params.append(min_score)

    if query:
        sql += " AND (title LIKE ? OR company LIKE ?)"
        params.append(f"%{query}%")
        params.append(f"%{query}%")

    if today_only:
        sql += " AND date(scraped_at) = date('now')"

    if sort_by == "score":
        sql += " ORDER BY similarity_score DESC"
    else:
        sql += " ORDER BY scraped_at DESC"

    sql += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(sql, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ─── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats():
    stats = get_db_stats()
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()

    return {
        **stats,
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(mem.percent, 1),
        "ram_used_gb": round(mem.used / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/jobs")
def api_jobs(limit: int = 10, offset: int = 0, status: str = None, min_score: float = None, query: str = None, sort_by: str = "scraped_at", today_only: bool = False):
    jobs = get_recent_jobs(limit=limit, offset=offset, status=status, min_score=min_score, query=query, sort_by=sort_by, today_only=today_only)
    total = count_jobs(status=status, min_score=min_score, query=query, today_only=today_only)
    return {"jobs": jobs, "total": total}


@app.get("/api/pipeline-log")
def api_pipeline_log(limit: int = 10):
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        logs = json.load(f)
    return logs[-limit:]


@app.get("/api/apply-log")
def api_apply_log(limit: int = 20):
    if not APPLY_LOG.exists():
        return []
    with open(APPLY_LOG) as f:
        logs = json.load(f)
    return logs[-limit:]


@app.post("/api/run-pipeline")
def api_run_pipeline(background_tasks: BackgroundTasks):
    """Trigger a pipeline run in the background."""
    def run():
        import subprocess
        subprocess.run(["python", "main.py", "--once"],
                       cwd=str(Path(__file__).parent))

    background_tasks.add_task(run)
    return {"message": "Pipeline started in background"}


@app.get("/api/ollama-status")
def api_ollama_status():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}


# ─── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Job Agent — Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1224 100%);
    border-bottom: 1px solid #2d3748;
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .badge {
    background: #10b981;
    color: white;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  main { padding: 32px; max-width: 1400px; margin: 0 auto; }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .stat-card {
    background: linear-gradient(135deg, #1a2035 0%, #141928 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
  }
  .stat-card:hover { transform: translateY(-2px); }
  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg, #667eea, #764ba2));
    border-radius: 999px 999px 0 0;
  }
  .stat-card .label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .stat-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 8px 0 4px;
    line-height: 1;
  }
  .stat-card .sub { color: #64748b; font-size: 0.8rem; }
  .section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
  }
  .table-card {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 32px;
  }
  .table-card table { width: 100%; border-collapse: collapse; }
  .table-card th {
    background: #141928;
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 12px 16px;
    text-align: left;
  }
  .table-card td {
    padding: 12px 16px;
    border-top: 1px solid #2d3748;
    font-size: 0.875rem;
    color: #cbd5e1;
  }
  .table-card tr:hover td { background: #1e293b; }
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
  }
  .pill-green { background: #064e3b; color: #10b981; }
  .pill-blue { background: #1e3a5f; color: #60a5fa; }
  .pill-yellow { background: #3f3108; color: #fbbf24; }
  .pill-gray { background: #374151; color: #9ca3af; }
  .pill-red { background: #4f0000; color: #f87171; }
  .actions { margin-bottom: 24px; display: flex; gap: 12px; }
  button.btn {
    padding: 10px 24px;
    border-radius: 10px;
    border: none;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
  }
  .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-secondary {
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #2d3748;
  }
  .btn-secondary:hover { background: #2d3748; color: white; }
  .system-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 32px;
  }
  .progress-bar {
    height: 8px;
    background: #2d3748;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 8px;
  }
  .progress-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.5s ease;
  }
  .ollama-badge { display: flex; align-items: center; gap: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-green { background: #10b981; }
  .dot-red { background: #ef4444; }
  #notification {
    position: fixed; top: 20px; right: 20px;
    background: #10b981; color: white;
    padding: 12px 24px; border-radius: 12px;
    font-weight: 600; display: none;
    animation: slideIn 0.3s ease;
    z-index: 999;
  }
  @keyframes slideIn {
    from { transform: translateX(100px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
</style>
</head>
<body>

<header>
  <h1>🤖 AI Job Agent Dashboard</h1>
  <span class="badge" id="status-badge">● LIVE</span>
</header>

<main>

  <div class="stats-grid" id="stats-grid">
    <div class="stat-card" style="--accent: linear-gradient(90deg, #667eea, #764ba2)">
      <div class="label">Jobs Scraped</div>
      <div class="value" id="total-scraped">—</div>
      <div class="sub">Total in database</div>
    </div>
    <div class="stat-card" style="--accent: linear-gradient(90deg, #06b6d4, #3b82f6)">
      <div class="label">Matched (≥60%)</div>
      <div class="value" id="total-matched">—</div>
      <div class="sub">Relevant to your profile</div>
    </div>
    <div class="stat-card" style="--accent: linear-gradient(90deg, #10b981, #34d399)">
      <div class="label">Applications Sent</div>
      <div class="value" id="total-applied">—</div>
      <div class="sub" id="today-applied-sub">— today</div>
    </div>
    <div class="stat-card" style="--accent: linear-gradient(90deg, #f59e0b, #ef4444)">
      <div class="label">CPU Usage</div>
      <div class="value" id="cpu-value">—</div>
      <div class="sub">Mac M2 load</div>
      <div class="progress-bar">
        <div class="progress-fill" id="cpu-bar" style="width:0%"></div>
      </div>
    </div>
    <div class="stat-card" style="--accent: linear-gradient(90deg, #8b5cf6, #ec4899)">
      <div class="label">RAM Usage</div>
      <div class="value" id="ram-value">—</div>
      <div class="sub" id="ram-sub">— GB used</div>
      <div class="progress-bar">
        <div class="progress-fill" id="ram-bar" style="width:0%"></div>
      </div>
    </div>
    <div class="stat-card" style="--accent: linear-gradient(90deg, #06b6d4, #10b981)">
      <div class="label">Ollama LLM</div>
      <div class="value" style="font-size:1.2rem" id="ollama-status">Checking...</div>
      <div class="sub" id="ollama-models">—</div>
    </div>
  </div>

  <div class="actions" style="flex-wrap: wrap; gap: 16px;">
    <div style="display: flex; gap: 8px;">
      <button class="btn btn-primary" onclick="runPipeline()">▶ Run Pipeline Now</button>
      <button class="btn btn-secondary" onclick="loadJobs(0)">🔄 Refresh</button>
    </div>
    
    <div style="flex-grow: 1; display: flex; gap: 8px; min-width: 300px;">
      <input type="text" id="search-box" class="btn btn-secondary" placeholder="Search title or company..." style="flex-grow: 1; text-align: left;" oninput="debounceLoad()">
    </div>
    
    <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
      <select id="status-filter" class="btn btn-secondary" onchange="loadJobs(0)" style="appearance: auto; cursor: pointer;">
        <option value="">Status: All</option>
        <option value="new">New</option>
        <option value="tailored">Tailored</option>
        <option value="applied">Applied</option>
        <option value="low_match">Low Match</option>
        <option value="manual_review">Review</option>
      </select>
      
      <select id="score-filter" class="btn btn-secondary" onchange="loadJobs(0)" style="appearance: auto; cursor: pointer;">
        <option value="">Score: All</option>
        <option value="0.75">High (75%+)</option>
        <option value="0.60">Match (60%+)</option>
      </select>

      <label class="btn btn-secondary" style="display: flex; gap: 6px; align-items: center; cursor: pointer;">
        <input type="checkbox" id="today-filter" onchange="loadJobs(0)"> Today
      </label>
      
      <select id="sort-by" class="btn btn-secondary" onchange="loadJobs(0)" style="appearance: auto; cursor: pointer;">
        <option value="scraped_at">Sort: Recent</option>
        <option value="score">Sort: Best Match</option>
      </select>
      
      <button class="btn btn-secondary" style="color:#ef4444" onclick="clearFilters()">Clear</button>
    </div>
  </div>

  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <p class="section-title" style="margin-bottom: 0;">Jobs (<span id="job-count">0</span>)</p>
    <div id="pagination-controls" style="display: flex; gap: 8px; align-items: center;">
       <button class="btn btn-secondary" id="prev-page" onclick="changePage(-1)" style="padding: 4px 12px;">←</button>
       <span id="page-info" style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Page 1</span>
       <button class="btn btn-secondary" id="next-page" onclick="changePage(1)" style="padding: 4px 12px;">→</button>
    </div>
  </div>

  <div class="table-card">
    <table>
      <thead>
        <tr>
          <th>Title</th>
          <th>Company</th>
          <th>Location</th>
          <th>Match %</th>
          <th>Status</th>
          <th>Date</th>
        </tr>
      </thead>
      <tbody id="jobs-table">
        <tr><td colspan="6" style="text-align:center;color:#64748b">Loading...</td></tr>
      </tbody>
    </table>
  </div>

</main>

<div id="notification"></div>

<script>
  function showNotif(msg, color='#10b981') {
    const n = document.getElementById('notification');
    n.style.background = color;
    n.textContent = msg;
    n.style.display = 'block';
    setTimeout(() => n.style.display = 'none', 3000);
  }

  function statusPill(status) {
    const map = {
      'new': ['pill-gray', 'NEW'],
      'tailored': ['pill-blue', 'TAILORED'],
      'applied': ['pill-green', 'APPLIED'],
      'low_match': ['pill-red', 'LOW MATCH'],
      'manual_review': ['pill-yellow', 'REVIEW'],
    };
    const [cls, label] = map[status] || ['pill-gray', status.toUpperCase()];
    return `<span class="pill ${cls}">${label}</span>`;
  }

  function scoreColor(score) {
    if (score >= 0.75) return '#10b981';
    if (score >= 0.60) return '#f59e0b';
    return '#94a3b8';
  }

  async function loadStats() {
    try {
      const r = await fetch('/api/stats');
      const d = await r.json();

      document.getElementById('total-scraped').textContent = d.total_scraped;
      document.getElementById('total-matched').textContent = d.total_matched;
      document.getElementById('total-applied').textContent = d.total_applied;
      document.getElementById('today-applied-sub').textContent = `${d.today_applied}/5 today`;
      document.getElementById('cpu-value').textContent = d.cpu_percent + '%';
      document.getElementById('cpu-bar').style.width = d.cpu_percent + '%';
      document.getElementById('ram-value').textContent = d.ram_percent + '%';
      document.getElementById('ram-bar').style.width = d.ram_percent + '%';
      document.getElementById('ram-sub').textContent = `${d.ram_used_gb} / ${d.ram_total_gb} GB`;

      const ollamaR = await fetch('/api/ollama-status');
      const ollamaD = await ollamaR.json();
      const ollamaEl = document.getElementById('ollama-status');
      const ollamaModels = document.getElementById('ollama-models');
      if (ollamaD.running) {
        ollamaEl.innerHTML = '<span style="color:#10b981">● Running</span>';
        ollamaModels.textContent = ollamaD.models.slice(0, 2).join(', ') || 'No models';
      } else {
        ollamaEl.innerHTML = '<span style="color:#ef4444">● Offline</span>';
        ollamaModels.textContent = 'Run: ollama serve';
      }
    } catch(e) {
      showNotif('Stats refresh failed', '#ef4444');
    }
  }

  let currentPage = 0;
  const PAGE_SIZE = 10;
  let debounceTimer;

  function debounceLoad() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => loadJobs(0), 400);
  }

  function changePage(delta) {
    currentPage = Math.max(0, currentPage + delta);
    loadJobs(currentPage);
  }

  function clearFilters() {
    document.getElementById('status-filter').value = '';
    document.getElementById('score-filter').value = '';
    document.getElementById('search-box').value = '';
    document.getElementById('today-filter').checked = false;
    document.getElementById('sort-by').value = 'scraped_at';
    loadJobs(0);
  }

  async function loadJobs(page = 0) {
    currentPage = page;
    try {
      const status = document.getElementById('status-filter').value;
      const minScore = document.getElementById('score-filter').value;
      const query = document.getElementById('search-box').value;
      const sortBy = document.getElementById('sort-by').value;
      const today = document.getElementById('today-filter').checked;
      
      let url = `/api/jobs?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`;
      if (status) url += `&status=${status}`;
      if (minScore) url += `&min_score=${minScore}`;
      if (query) url += `&query=${encodeURIComponent(query)}`;
      if (sortBy) url += `&sort_by=${sortBy}`;
      if (today) url += `&today_only=true`;

      const r = await fetch(url);
      const data = await r.json();
      const jobs = data.jobs;
      const total = data.total;
      
      const tbody = document.getElementById('jobs-table');
      document.getElementById('job-count').textContent = total;
      document.getElementById('page-info').textContent = `Page ${page + 1} of ${Math.ceil(total/PAGE_SIZE) || 1}`;

      if (!jobs.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#64748b;padding:40px">No matching jobs found.</td></tr>';
        return;
      }

      tbody.innerHTML = jobs.map(j => `
        <tr>
          <td>
            <div style="font-weight:600;color:#f8fafc">${j.title}</div>
            <a href="${j.url || '#'}" target="_blank" style="color:#60a5fa;text-decoration:none;font-size:0.75rem">View Listing →</a>
          </td>
          <td>${j.company}</td>
          <td>${j.location || '—'}</td>
          <td style="color:${scoreColor(j.similarity_score)};font-weight:700">
            ${(j.similarity_score * 100).toFixed(0)}%
          </td>
          <td>${statusPill(j.status)}</td>
          <td style="color:#64748b;font-size:0.75rem">
            ${new Date(j.scraped_at).toLocaleDateString()}<br>
            ${new Date(j.scraped_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
          </td>
        </tr>
      `).join('');
    } catch(e) {
      showNotif('Jobs refresh failed', '#ef4444');
    }
  }

  async function runPipeline() {
    showNotif('🚀 Starting pipeline...', '#667eea');
    try {
      await fetch('/api/run-pipeline', { method: 'POST' });
      showNotif('✅ Pipeline started in background!');
    } catch(e) {
      showNotif('❌ Failed to start pipeline', '#ef4444');
    }
  }

  // Initial load
  loadStats();
  loadJobs();

  // Auto-refresh every 30s
  setInterval(() => {
    loadStats();
    loadJobs();
  }, 30000);
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


if __name__ == "__main__":
    print("🚀 Starting AI Job Agent Dashboard")
    print("   Open: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
