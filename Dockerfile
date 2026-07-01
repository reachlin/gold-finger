FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps — installed before copying source so layer is cached
COPY requirements-overseer.txt .
RUN pip install --no-cache-dir -r requirements-overseer.txt

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
