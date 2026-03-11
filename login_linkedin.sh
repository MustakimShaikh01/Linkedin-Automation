#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# LinkedIn Login — One Click
# This opens a real Chrome browser. Log in with YOUR details.
# Cookies are saved automatically. No password stored in code.
# ─────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   🔐 LinkedIn Session Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${YELLOW}What will happen:${NC}"
echo "  1. A Chrome browser window will open"
echo "  2. LinkedIn login page will appear"
echo "  3. YOU type your email + password in the browser"
echo "  4. Complete any 2FA verification if asked"
echo "  5. Script saves your session automatically"
echo "  6. Bot runs headless from now on (no more logins)"
echo ""
echo -e "  ${YELLOW}Your password is NEVER saved to any file.${NC}"
echo -e "  ${YELLOW}Only browser cookies are stored locally.${NC}"
echo ""
echo -e "  Press ENTER to open the browser, or Ctrl+C to cancel."
read -r

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "  ⚠️  Run setup.sh first: chmod +x setup.sh && ./setup.sh"
    exit 1
fi

# Run the session manager
python automation/linkedin_session.py --login

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check result
if [ -f "database/sessions/linkedin_cookies.json" ]; then
    echo -e "${GREEN}  ✅ LinkedIn session saved!${NC}"
    echo ""
    echo "  To check status:   python automation/linkedin_session.py --check"
    echo "  To run pipeline:   python main.py --once"
    echo "  To see dashboard:  python dashboard.py"
else
    echo -e "  ❌ Session not saved. Please try again."
    echo "  Make sure you completed login in the browser window."
fi
echo ""
