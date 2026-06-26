"""
Daily market intelligence for NVDA:
- Recent news headlines (Finnhub + RSS feeds)
- Technical trend summary (ta library)
- Sentiment score
"""
import os
import sys
import time
import feedparser
import finnhub
import pandas as pd
import ta
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

FINNHUB_TOKEN = os.environ.get("FINNHUB_API_KEY", "")

RSS_FEEDS = {
    "Reuters":  "https://feeds.reuters.com/reuters/businessNews",
    "SeekAlpha": "https://seekingalpha.com/api/sa/combined/NVDA.xml",
    "Yahoo":    "https://finance.yahoo.com/rss/headline?s=NVDA",
}

KEYWORDS = ["nvidia", "nvda", "gpu", "ai chip", "data center", "jensen huang", "blackwell", "hopper"]


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
def fetch_rss_news(symbol="NVDA", max_per_feed=5):
    headlines = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "")
                if any(kw in title.lower() for kw in KEYWORDS):
                    headlines.append({
                        "source":    source,
                        "title":     title,
                        "published": entry.get("published", ""),
                        "link":      entry.get("link", ""),
                    })
        except Exception:
            pass
    return headlines


def fetch_finnhub_news(symbol="NVDA", days=3):
    if not FINNHUB_TOKEN:
        return []
    client = finnhub.Client(api_key=FINNHUB_TOKEN)
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        news = client.company_news(symbol, _from=start, to=end)
        return [{"source": n["source"], "title": n["headline"],
                 "published": datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M"),
                 "link": n["url"]} for n in news[:10]]
    except Exception:
        return []


def score_sentiment(headlines):
    """Simple keyword-based sentiment: +1 bullish, -1 bearish words."""
    bullish = ["surge", "beat", "record", "rally", "upgrade", "buy", "outperform",
               "strong", "growth", "demand", "partnership", "contract", "win"]
    bearish = ["fall", "drop", "miss", "downgrade", "sell", "underperform", "weak",
               "loss", "cut", "concern", "tariff", "ban", "competition", "slump"]
    score = 0
    for h in headlines:
        text = h["title"].lower()
        score += sum(1 for w in bullish if w in text)
        score -= sum(1 for w in bearish if w in text)
    return score


# ---------------------------------------------------------------------------
# Technical trend analysis
# ---------------------------------------------------------------------------
def analyze_trend(df: pd.DataFrame) -> dict:
    """Run ta indicators and return a human-readable trend summary."""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    rsi   = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd  = ta.trend.MACD(close)
    macd_hist = macd.macd_diff().iloc[-1]
    bb    = ta.volatility.BollingerBands(close, window=20)
    bb_pctb = bb.bollinger_pband().iloc[-1]
    atr   = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    adx   = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
    obv   = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    obv_trend = "rising" if obv.iloc[-1] > obv.iloc[-5] else "falling"

    price = close.iloc[-1]

    trend = "BULLISH" if ema20 > ema50 else "BEARISH"
    momentum = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
    bb_pos = "upper band" if bb_pctb > 0.8 else ("lower band" if bb_pctb < 0.2 else "middle")

    return {
        "price":      round(price, 2),
        "trend":      trend,
        "ema20":      round(ema20, 2),
        "ema50":      round(ema50, 2),
        "rsi":        round(rsi, 1),
        "momentum":   momentum,
        "macd_hist":  round(macd_hist, 4),
        "bb_pctb":    round(bb_pctb, 2),
        "bb_pos":     bb_pos,
        "atr":        round(atr, 2),
        "adx":        round(adx, 1),
        "obv_trend":  obv_trend,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(symbol, tech, headlines, sentiment_score):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  {symbol} Market Intelligence  [{now}]")
    print(f"{'='*60}")

    print(f"\n[TECHNICALS]  Price: ${tech['price']}  |  Trend: {tech['trend']}")
    print(f"  EMA20={tech['ema20']}  EMA50={tech['ema50']}  (price {'above' if tech['price'] > tech['ema50'] else 'below'} 50-EMA)")
    print(f"  RSI={tech['rsi']} ({tech['momentum']})  |  MACD hist={tech['macd_hist']}")
    print(f"  BB %B={tech['bb_pctb']} ({tech['bb_pos']})  |  ATR={tech['atr']}  |  ADX={tech['adx']}")
    print(f"  OBV: {tech['obv_trend']}")

    sig = "BULLISH" if sentiment_score > 0 else ("BEARISH" if sentiment_score < 0 else "NEUTRAL")
    print(f"\n[NEWS SENTIMENT]  Score: {sentiment_score:+d}  →  {sig}")
    if headlines:
        print(f"  Recent headlines ({len(headlines)}):")
        for h in headlines[:6]:
            print(f"    [{h['source']}] {h['title'][:80]}")
    else:
        print("  No matching headlines found.")

    print(f"\n[OVERALL SIGNAL]")
    bull_signals = sum([
        tech["trend"] == "BULLISH",
        tech["rsi"] < 60,
        tech["macd_hist"] > 0,
        tech["obv_trend"] == "rising",
        sentiment_score > 0,
    ])
    total = 5
    print(f"  {bull_signals}/{total} bullish signals  →  {'BUY BIAS' if bull_signals >= 3 else 'SELL/HOLD BIAS'}")
    print(f"{'='*60}\n")


def run(df: pd.DataFrame, symbol="NVDA"):
    print(f"Analyzing {symbol}...")
    tech = analyze_trend(df)

    print("Fetching news...")
    headlines = fetch_finnhub_news(symbol) or fetch_rss_news(symbol)
    sentiment = score_sentiment(headlines)

    print_report(symbol, tech, headlines, sentiment)
    return tech, headlines, sentiment


if __name__ == "__main__":
    import schwab as schwab_lib
    from schwab.nvda_trader import get_client, fetch_nvda_history, candles_to_df
    sys.path.insert(0, os.path.dirname(__file__))
    from nvda_trader import get_client, fetch_nvda_history, candles_to_df

    client = get_client()
    candles = fetch_nvda_history(client)
    df = candles_to_df(candles)
    run(df)
