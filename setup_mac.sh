#!/bin/bash
# One-time setup on macOS. Run from VS Code terminal:  bash setup_mac.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo ""
echo "Done. Next: open .env, paste your Groq key, then press F5 (or run: streamlit run app.py)"
