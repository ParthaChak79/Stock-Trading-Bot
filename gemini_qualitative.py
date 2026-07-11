"""
================================================================================
 GEMINI QUALITATIVE SCORING
================================================================================
Supplies the *qualitative* criterion scores that STOCK_SCREENER_SPEC.md leaves
to researched judgment (moat, management, governance, AI exposure, etc.), using
Google Gemini 3.1 Pro with Google-Search grounding, called over the REST API
(generativelanguage.googleapis.com) so no extra SDK dependency is required.

Design decisions (see the CLAUDE.md discussion that led here):
  - Quantitative criteria are re-scored from fresh TradingView data every run, but
    qualitative judgments don't move week to week and re-running the LLM weekly
    would reshuffle ranks on model noise alone. So qualitative scores are CACHED
    per ticker in qualitative_scores.json and only recomputed when missing or
    older than CACHE_TTL_DAYS. New names always get fresh scores.
  - GRACEFUL DEGRADATION: if GEMINI_API_KEY is unset or the call/parse fails, we
    return {} — the scoring engine then falls back to the spec's per-criterion
    defaults. The Sunday job never crashes on a missing key or an API hiccup.

Set GEMINI_API_KEY in .env (free/paid key from https://aistudio.google.com/apikey).
================================================================================
"""

import os
import re
import json
from datetime import datetime, timezone, timedelta

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "qualitative_scores.json")

# Model is resolved at call time (not import) so a GEMINI_MODEL override in .env
# is picked up after dotenv loads. Default is Gemini 3.1 Pro.
DEFAULT_MODEL = "gemini-3.1-pro-preview"
CACHE_TTL_DAYS = 30


def _model():
    return _clean_env_var(os.getenv("GEMINI_MODEL")) or DEFAULT_MODEL


def _gemini_url():
    return (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{_model()}:generateContent")

# Numeric qualitative fields and their valid ranges (min, max).
# Everything not listed is treated as 0-100.
_SUBSCORE_RANGES = {
    "ai_revenue_vuln": (0, 40),
    "ai_adaptability": (0, 35),
    "ai_demand_tailwind": (0, 25),
}
_QUAL_FIELDS = [
    # Category A qualitative
    "asset_light", "equity_dilution_history", "working_capital_trend",
    "dividend_consistency", "customer_concentration",
    # Category B qualitative
    "forward_growth_potential",
    # Category C qualitative
    "economic_moat", "market_leadership", "addressable_market",
    "sector_growth", "revenue_diversification", "biz_model_quality",
    # Category D qualitative
    "corporate_governance", "promoter_holding", "institutional_ownership",
    "insider_trading", "longevity_prior",
    # Category F qualitative
    "sector_tailwinds_headwinds", "active_macro_event",
    # AI exposure sub-scores (ranges above)
    "ai_revenue_vuln", "ai_adaptability", "ai_demand_tailwind",
]


def _clean_env_var(value):
    if not value:
        return value
    return value.split("#")[0].strip()


def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _is_fresh(entry):
    asof = entry.get("asof")
    if not asof:
        return False
    try:
        d = datetime.strptime(asof, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - d) <= timedelta(days=CACHE_TTL_DAYS)


def _build_prompt(ticker, data):
    """Prompt Gemini with the live quantitative context + the spec's rubrics, and
    demand a strict JSON object of qualitative scores with reasoning."""
    ctx = {k: data.get(k) for k in
           ("name", "sector", "industry", "roe", "roce", "pe", "de", "mcap",
            "sg1y", "sg5y", "pg1y", "pg5y", "ltp")}
    return f"""You are a fundamental equity analyst scoring the Indian (NSE) stock \
{ticker} ({data.get('name')}). Research it using current information and return \
ONLY a JSON object (no prose, no markdown fences) with the fields below.

Live quantitative context (already computed elsewhere — do NOT re-score these):
{json.dumps(ctx, indent=2)}

Score each field 0-100 unless a different range is stated. Base every score on \
real, researched facts about THIS company (moat, management, ownership, sector \
dynamics). If you genuinely cannot assess a field, use 50 (or the midpoint of \
its range). Never fabricate specifics.

Fields (all integers):
  "sector_label": a concise sector label string (e.g. "Pharmaceuticals", \
"Defense", "Automobiles", "Banks", "Healthcare", "Industrials", "Power").

  Category A (quality) qualitative:
  "asset_light", "equity_dilution_history", "working_capital_trend", \
"dividend_consistency", "customer_concentration"

  Category B (valuation) qualitative:
  "forward_growth_potential"

  Category C (business) qualitative:
  "economic_moat", "market_leadership", "addressable_market", "sector_growth", \
"revenue_diversification", "biz_model_quality"

  Category D (governance) qualitative:
  "corporate_governance", "promoter_holding", "institutional_ownership", \
"insider_trading", "longevity_prior"

  Category F (momentum/catalysts) qualitative:
  "sector_tailwinds_headwinds", "active_macro_event"

  AI / Automation net exposure — three researched sub-scores (NOT 0-100):
  "ai_revenue_vuln" (0-40): revenue-model vulnerability to AI displacement. \
40=near-zero risk (physical assets, regulated monopoly, branded physical goods, \
defence hardware); 32=low (pharma, hospitals, branded consumer, auto OEMs); \
24=moderate (banks/NBFCs, engineering/EPC); 16=high (headcount IT services); \
8=very high (pure labour arbitrage).
  "ai_adaptability" (0-35): 35=AI-native/platform; 28=active adaptor; \
21=moderate; 14=passive; 7=laggard.
  "ai_demand_tailwind" (0-25): 25=strong pull (defence AI, data-centre \
power/cabling, electronics mfg, semi-adjacent chemicals); 20=moderate \
(healthcare AI, fintech, telecom infra); 15=neutral (autos, pharma generics, \
cement); 10=mild headwind (commodity IT); 5=strong headwind (BPO).

  "reasoning": one short string summarising the key judgments (moat, governance, \
AI exposure).

Return the JSON object only."""


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _clamp(parsed):
    """Keep only known fields, coerce to int, clamp to each field's valid range."""
    out = {}
    if isinstance(parsed.get("sector_label"), str):
        out["sector_label"] = parsed["sector_label"].strip()
    if isinstance(parsed.get("reasoning"), str):
        out["reasoning"] = parsed["reasoning"].strip()
    for f in _QUAL_FIELDS:
        v = parsed.get(f)
        if v is None:
            continue
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            continue
        lo, hi = _SUBSCORE_RANGES.get(f, (0, 100))
        out[f] = max(lo, min(hi, v))
    return out


def _call_gemini(api_key, prompt):
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2},
    }
    resp = requests.post(_gemini_url(), params={"key": api_key},
                         json=body, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def get_qualitative(ticker_norm, data, cache, api_key, force=False):
    """Return a qualitative-scores dict for one stock. Uses the cache unless the
    entry is stale/missing/forced. Falls back to {} (-> spec defaults) if Gemini
    is unavailable or the response can't be parsed. `cache` is mutated in place;
    the caller is responsible for persisting it via save_cache()."""
    entry = cache.get(ticker_norm)
    if entry and _is_fresh(entry) and not force:
        return entry.get("scores", {})

    if not api_key:
        return entry.get("scores", {}) if entry else {}

    try:
        text = _call_gemini(api_key, _build_prompt(ticker_norm, data))
        parsed = _extract_json(text)
        if not parsed:
            print(f"[gemini] {ticker_norm}: could not parse JSON; using defaults")
            return entry.get("scores", {}) if entry else {}
        scores = _clamp(parsed)
        cache[ticker_norm] = {
            "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "name": data.get("name"),
            "scores": scores,
        }
        return scores
    except Exception as e:
        print(f"[gemini] {ticker_norm}: call failed ({e}); using {'cached' if entry else 'defaults'}")
        return entry.get("scores", {}) if entry else {}


def load_api_key():
    return _clean_env_var(os.getenv("GEMINI_API_KEY"))


def load_cache():
    return _load_cache()


def save_cache(cache):
    _save_cache(cache)
