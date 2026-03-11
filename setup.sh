#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AI Job Agent — One-click Setup for Mac M2
# Run: chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   🤖 AI Job Agent — Mac M2 Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ─── 1. Python version check ─────────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "  ${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "  ${RED}✗ Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    echo "  Install from: https://www.python.org/downloads/"
    exit 1
fi

# ─── 2. Virtual environment ───────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Setting up virtual environment...${NC}"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "  ${GREEN}✓ Created .venv${NC}"
else
    echo -e "  ${GREEN}✓ .venv already exists${NC}"
fi

source .venv/bin/activate
echo -e "  ${GREEN}✓ Activated .venv${NC}"

# ─── 3. Install dependencies ──────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "  ${GREEN}✓ Dependencies installed${NC}"

# ─── 4. Install Playwright browsers ──────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Installing Playwright (headless Chrome)...${NC}"
playwright install chromium
echo -e "  ${GREEN}✓ Playwright chromium installed${NC}"

# ─── 5. Create directory structure ───────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Creating directory structure...${NC}"

mkdir -p database embeddings resume/tailored llm/qlora_dataset database/manual_review

# Create __init__.py files for module imports
touch scraper/__init__.py
touch matcher/__init__.py
touch llm/__init__.py
touch automation/__init__.py

echo -e "  ${GREEN}✓ Directories created${NC}"

# ─── 6. Check Ollama ──────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Checking Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "  ${GREEN}✓ Ollama is installed${NC}"

    # Start Ollama in background if not running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "  Starting Ollama server..."
        ollama serve &
        sleep 3
    fi

    # Check for models
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | wc -l | xargs)
    if [ "$MODELS" -eq "0" ]; then
        echo ""
        echo -e "  ${YELLOW}⚠ No models found. Pulling Mistral 7B (recommended)...${NC}"
        echo "  This may take 5-10 minutes on first run."
        ollama pull mistral
        echo -e "  ${GREEN}✓ Mistral 7B downloaded${NC}"
    else
        echo -e "  ${GREEN}✓ $MODELS model(s) available${NC}"
        ollama list
    fi
else
    echo -e "  ${YELLOW}⚠ Ollama not found. Install from: https://ollama.ai${NC}"
    echo "  After installing, run: ollama pull mistral"
fi

# ─── Build FAISS index ────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Building resume FAISS index...${NC}"
python3 -c "
from matcher.similarity_engine import build_resume_faiss_index
build_resume_faiss_index()
print('FAISS index built successfully!')
" && echo -e "  ${GREEN}✓ FAISS index ready${NC}" || echo -e "  ${YELLOW}⚠ Index will be built on first run${NC}"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅ Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}Quick Start:${NC}"
echo ""
echo "  # Activate environment"
echo "  source .venv/bin/activate"
echo ""
echo "  # Start the dashboard"
echo "  python dashboard.py"
echo "  → Open http://localhost:8000"
echo ""
echo "  # Run pipeline once (test)"
echo "  python main.py --once"
echo ""
echo "  # Run pipeline on schedule (every 6 hours)"
echo "  python main.py"
echo ""
echo "  # Generate QLoRA dataset"
echo "  python llm/qlora_dataset_generator.py"
echo ""
echo "  # Test similarity engine"
echo "  python matcher/similarity_engine.py"
echo ""
