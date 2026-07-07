FROM python:3.11-slim

WORKDIR /app

# System deps (libgomp1 is a lightgbm runtime dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps — installed before copying source so layer is cached
COPY requirements-overseer.txt .
RUN pip install --no-cache-dir -r requirements-overseer.txt

# ── Optional: full advisory stack in the container ──────────────────────────
# The TimesFM forecasts (timesfm_30d_pct + the wheel-vs-hold router) need
# torch + timesfm, which are host installs today — WITHOUT these lines the
# containerized scanner runs with TimesFM/Kronos gracefully disabled and the
# router never fires. Uncomment to bake them in. Costs: image grows ~2.5GB
# (CPU torch), and the 200M checkpoint (~1GB) downloads from HuggingFace on
# first start — mount a cache volume to keep it across container recreates:
#   volumes: - ./hf-cache:/root/.cache/huggingface
# (Kronos would additionally need its repo cloned into the image — separate.)
#
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir "timesfm @ git+https://github.com/google-research/timesfm"
# ENV TIMESFM_SRC=/usr/local/lib/python3.11/site-packages
# ─────────────────────────────────────────────────────────────────────────────

# Copy project source
COPY . .

# Persistent volumes:
#   /app/data             — paper ledger, trades JSON, fast_riskoff state
#   /app/schwab           — schwab_token.json (Schwab OAuth token, refreshed at runtime)
VOLUME ["/app/data", "/app/schwab"]

# Default: paper mode with Anthropic (override via ENV or CMD args)
#   LLM_PROVIDER   = anthropic | openai | deepseek | openai_compatible
#   LLM_MODEL      = model name
#   LLM_API_KEY    = API key (or use provider-specific: ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
#   LLM_BASE_URL   = for openai_compatible endpoints
#   REALLY_REAL    = true  → enable real Schwab order placement (dangerous!)
#   SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET = required for Schwab API
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/schwab:/app

CMD ["python", "schwab/auto_overseer.py", "--paper"]
