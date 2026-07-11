# NSE Stock Screener — Algorithm Specification

## Purpose

This document specifies a complete fundamental scoring engine for Indian (NSE-listed)
equities. Implement this exactly as written. It should let you score any stock — new or
already tracked — pull the required data from the `tapetide` MCP server, and determine
whether it qualifies for inclusion in a ranked investment list, without needing further
clarification from the user.

## Investment philosophy this encodes

The goal is a rolling, self-refreshing list of high-quality, growing businesses in
expanding sectors. This scoring engine is the fundamental quality gate in that process —
a conviction filter, not a simple screener. A stock only qualifies if it clears **both**
a hard exclusion filter (business model / sector / growth profile) **and** a minimum
score threshold.

**Inclusion rule — a stock is added to the ranked list if and only if:**
1. It is NOT in an excluded sector (Section 3)
2. It is NOT flagged as a stagnant-growth business (Section 4)
3. Its final weighted score is **≥ 60.0** (out of 100)

---

## 1. Data source — tapetide MCP

For each stock:

```
tapetide.search_stocks(query="<company name>")   # resolve NSE symbol + sector if unknown
tapetide.get_company_profile(symbol="<NSE_SYMBOL>", include=["ratings"])
```

### Fields to extract into a flat dict per stock

| Field | Source path (typical) | Notes |
|---|---|---|
| `roe` | `fundamentals.roe` | Return on Equity, % |
| `roce` | `fundamentals.roce` | Return on Capital Employed, % |
| `pe` | `valuation.pe_ttm` or `fundamentals.stock_pe` | Trailing P/E |
| `pb` | `fundamentals.price_to_book` | Price/Book |
| `de` | `fundamentals.debt_to_equity` | Debt/Equity ratio |
| `mcap` | `fundamentals.market_cap` | ₹ crores |
| `h52` | `fundamentals.high_52w` | 52-week high price |
| `ltp` | `quote.price` | Last traded price |
| `ebitda` | `fundamentals.ebitda` | ₹ crores |
| `bv` | `fundamentals.book_value` | Per share, ₹ |
| `issued` | `company.issued_size` | Shares outstanding |
| `ld` | `company.listing_date` | YYYY-MM-DD |
| `sg1y`, `sg3y`, `sg5y` | `unified_growth_rates.revenue_growth[...]` or `growth_metrics["Compounded Sales Growth"]` | Revenue CAGR, % |
| `pg1y`, `pg3y`, `pg5y` | `unified_growth_rates.net_income_growth[...]` or `growth_metrics["Compounded Profit Growth"]` | Profit CAGR, % |
| `buy` | `ratings.data.percent_buy` | Analyst % buy rating |
| `sector` | from `search_stocks` | Used for exclusion rules + sector-average lookups |
| `is_bank` | manual flag | `true` if sector == "Banks" |

If a field is unavailable, set it to `null`. Every scoring function below degrades
gracefully to a neutral default rather than failing on missing data — never fabricate a
plausible-looking number to fill a gap.

---

## 2. Categories, weights, and criteria

Five categories, weighted to sum to 100%:

| Category | Weight | What it measures |
|---|---|---|
| **A — Financial Quality** | 27.8% | Profitability, growth, balance sheet strength |
| **B — Valuation** | 14.2% | Is the price reasonable given growth/quality |
| **C — Business Fundamentals** | 27.8% | Moat, management, market position |
| **D — Governance & Ownership** | 11.1% | Promoter/institutional trust signals |
| **F — Momentum, Catalysts & AI Exposure** | 19.1% | Price action, sentiment, structural AI tailwind/risk |

```python
CATEGORY_WEIGHTS = {'A': .278, 'B': .142, 'C': .278, 'D': .111, 'F': .191}  # sums to 1.000

def final_score(catA, catB, catC, catD, catF):
    w = CATEGORY_WEIGHTS
    return round(catA*w['A'] + catB*w['B'] + catC*w['C'] + catD*w['D'] + catF*w['F'], 2)
```

Each category score is the weighted average of its available criteria — missing
criteria are excluded and remaining weights renormalized proportionally:

```python
def category_score(criterion_scores: dict, criterion_weights: dict) -> float:
    available = {k: v for k, v in criterion_scores.items() if v is not None}
    if not available:
        return 50.0  # neutral fallback if entire category is unscored
    total_weight = sum(criterion_weights[k] for k in available)
    return sum(available[k] * criterion_weights[k] / total_weight for k in available)
```

---

## 3. Category A — Financial Quality (8 criteria, weights sum to 100% of A)

| Criterion | Weight | Type |
|---|---|---|
| `capital_efficiency` | 20% | Quantitative |
| `revenue_earnings_growth` | 20% | Quantitative |
| `debt_quality` | 15% | Quantitative |
| `asset_light` | 12% | Qualitative |
| `equity_dilution_history` | 10% | Qualitative |
| `working_capital_trend` | 10% | Qualitative |
| `dividend_consistency` | 8% | Qualitative |
| `customer_concentration` | 5% | Qualitative |

```python
def score_capital_efficiency(roe, roce):
    rs = 50 if roe is None else (100 if roe>=30 else 90 if roe>=25 else 80 if roe>=20
         else 70 if roe>=15 else 50 if roe>=10 else 30)
    rcs = 50 if roce is None else (100 if roce>=25 else 85 if roce>=20 else 70 if roce>=15
          else 50 if roce>=10 else 30)
    return 0.4*rs + 0.4*rcs + 10

def score_revenue_earnings_growth(sg1y, sg3y, sg5y, pg1y, pg3y, pg5y):
    rS = 25 if sg1y is None else (50 if sg1y>=30 else 45 if sg1y>=20 else 40 if sg1y>=15
         else 30 if sg1y>=10 else 20 if sg1y>=5 else 10 if sg1y>0 else 0)
    s3 = 20 if sg3y is None else (30 if sg3y>=25 else 25 if sg3y>=20 else 20 if sg3y>=15 else 10)
    s5 = (20 if sg3y and sg3y>=20 else 15 if sg3y and sg3y>=15 else 15) if sg5y is None \
         else (20 if sg5y>=20 else 15 if sg5y>=15 else 10)
    eR = 25 if pg1y is None else (50 if pg1y>=30 else 45 if pg1y>=20 else 40 if pg1y>=15
         else 30 if pg1y>=10 else 20 if pg1y>=5 else 10 if pg1y>0 else 0)
    e3 = 20 if pg3y is None else (30 if pg3y>=25 else 25 if pg3y>=20 else 20 if pg3y>=15 else 10)
    fg = pg5y if pg5y is not None else pg3y
    e5 = 15 if fg is None else (20 if fg>=20 else 15 if fg>=15 else 10)
    return min(100, max(0, 0.6*(rS+s3+s5) + 0.4*(eR+e3+e5)))

def score_debt_quality(de, is_bank=False):
    if is_bank or de is None:
        return 50  # banks excluded from D/E scoring — use capital adequacy if available
    d_score = 50 if de<0.3 else 45 if de<0.5 else 35 if de<1.0 else 20 if de<2.0 else 5
    interest_proxy = 50 if de<0.3 else 40 if de<0.5 else 30 if de<1.0 else 20 if de<2.0 else 10
    return d_score + interest_proxy
```

`asset_light`, `equity_dilution_history`, `working_capital_trend`,
`dividend_consistency`, `customer_concentration` are qualitative. Research each one
(balance sheet trends, capex intensity, fundraising history, payout track record,
revenue mix) rather than defaulting blindly. If genuinely unresearched, default to
**50**.

---

## 4. Category B — Valuation (6 criteria, weights sum to 100% of B)

| Criterion | Weight | Type |
|---|---|---|
| `peg_ratio` | 25% | Quantitative |
| `pe_vs_sector` | 20% | Quantitative |
| `price_to_book` | 15% | Quantitative |
| `ev_ebitda` | 15% | Quantitative |
| `fcf_yield` | 15% | Qualitative/manual |
| `forward_growth_potential` | 10% | Qualitative |

```python
def score_peg(peg):
    if peg is None: return 50
    if peg < 0 or peg > 10: return 0
    return (100 if peg<0.2 else 95 if peg<0.5 else 85 if peg<0.8 else 75 if peg<1.0
            else 60 if peg<1.5 else 40 if peg<2.0 else 20)

def score_pe_vs_sector(pe, sector_avg_pe):
    if pe is None or pe <= 0:
        return 0 if pe is not None else 50
    baseline = sector_avg_pe or pe
    discount = (baseline - pe) / baseline * 100
    return (100 if discount>=50 else 90 if discount>=30 else 80 if discount>=20
            else 70 if discount>=10 else 60 if discount>=0 else 50 if discount>=-10
            else 35 if discount>=-20 else 20)

def score_price_to_book(pb):
    if pb is None: return 50
    if pb < 0: return 0
    return 100 if pb<1 else 85 if pb<2 else 65 if pb<3 else 45 if pb<5 else 25

def score_ev_ebitda(ev_ebitda, sector_avg_ev_ebitda):
    if ev_ebitda is None: return 50
    if not sector_avg_ev_ebitda: return 55
    discount = (sector_avg_ev_ebitda - ev_ebitda) / sector_avg_ev_ebitda * 100
    return (100 if discount>=40 else 85 if discount>=25 else 70 if discount>=10
            else 55 if discount>=0 else 40 if discount>=-15 else 20)

def compute_peg(pe, profit_growth_5y_or_3y):
    g = profit_growth_5y_or_3y
    if not pe or pe <= 0 or not g or g <= 0: return None
    return pe / min(g, 100)  # cap growth input to avoid absurd PEGs

def compute_ev_ebitda(market_cap, debt_to_equity, book_value_per_share, shares_outstanding, ebitda):
    if not all([market_cap, book_value_per_share, shares_outstanding, ebitda]) or ebitda <= 0:
        return None
    equity = book_value_per_share * shares_outstanding / 1e7  # ₹ crores
    debt = max(0, debt_to_equity or 0) * equity
    return (market_cap + debt) / ebitda
```

`fcf_yield` and `forward_growth_potential` are qualitative — default to **50** if
unresearched, but calculate FCF/market-cap directly where cash flow statement data is
available rather than leaving it neutral.

---

## 5. Category C — Business Fundamentals (7 criteria, weights sum to 100% of C)

| Criterion | Weight | Type |
|---|---|---|
| `economic_moat` | 22.2% | Qualitative |
| `management_quality` | 16.7% | Partially quantitative |
| `market_leadership` | 11.1% | Qualitative |
| `addressable_market` | 11.1% | Qualitative |
| `sector_growth` | 11.1% | Qualitative |
| `revenue_diversification` | 11.1% | Qualitative |
| `biz_model_quality` | 16.7% | Qualitative |

```python
def score_management_quality_proxy(roce):
    roce_component = 35 if roce is None else (50 if roce>=20 else 35 if roce>=15 else 15)
    return roce_component + 35  # floor 35, max 85 — refine with qualitative overlay
```

The remaining six criteria are qualitative. Actively research each — competitive
landscape, pricing power, market share trend, diversification of revenue streams,
business model durability — rather than leaving everything at a neutral default. Default
to **50** only when genuinely unresearched.

---

## 6. Category D — Governance & Ownership (5 criteria, weights sum to 100% of D)

| Criterion | Weight | Type |
|---|---|---|
| `corporate_governance` | 25% | Qualitative |
| `promoter_holding` | 25% | Qualitative/manual |
| `institutional_ownership` | 20% | Qualitative/manual |
| `insider_trading` | 15% | Qualitative/manual |
| `longevity_consistency` | 15% | Partially computed |

```python
def score_longevity(listing_date_str, qualitative_prior=50):
    """Blends listing-age stability signal with a qualitative consistency prior."""
    if not listing_date_str:
        return qualitative_prior
    from datetime import datetime
    try:
        age_years = (datetime.now() - datetime.strptime(listing_date_str, '%Y-%m-%d')).days / 365.25
    except Exception:
        return qualitative_prior
    age_score = (50 if age_years>=25 else 40 if age_years>=15 else 30 if age_years>=10
                 else 20 if age_years>=5 else 10)
    return round(0.4*age_score + 0.6*qualitative_prior)
```

For `promoter_holding`, `institutional_ownership`, `insider_trading`: if
`tapetide.get_shareholding` / `tapetide.get_promoter_pledge` are available and
authorized, use them directly (higher promoter holding with no pledge = higher score;
higher institutional ownership = higher score; clean insider-trading history = higher
score). Otherwise default to **50** and flag the stock's output as
`"governance_data": "estimated"`.

---

## 7. Category F — Momentum, Catalysts & AI Exposure (5 criteria, weights sum to 100% of F)

| Criterion | Weight | Type |
|---|---|---|
| `sector_tailwinds_headwinds` | 20% | Qualitative |
| `momentum` | 24% | Quantitative |
| `earnings_estimate_revisions` | 20% | Quantitative |
| `active_macro_event` | 16% | Qualitative |
| `ai_automation_net_exposure` | 20% | **Qualitative — see rubric below** |

```python
def score_momentum(ltp, high_52w):
    ratio = ltp / high_52w if (high_52w and ltp) else None
    base = 25 if ratio is None else (50 if ratio>=0.95 else 35 if ratio>=0.85
           else 20 if ratio>=0.70 else 5)
    return base + 20

def score_analyst_revisions(percent_buy):
    if percent_buy is None: return 50
    return (100 if percent_buy>=70 else 85 if percent_buy>=50 else 65 if percent_buy>=30
            else 40 if percent_buy>=15 else 15)
```

`sector_tailwinds_headwinds` and `active_macro_event` are qualitative — research current
sector-specific catalysts, policy shifts, and macro conditions rather than defaulting
blindly. Default to **50** only if genuinely unresearched.

### AI / Automation Net Exposure rubric

Score this criterion as the sum of three sub-scores, each researched and reasoned
through explicitly — this is the parameter most likely to be wrong if defaulted
blindly, since exposure varies enormously by business model even within the same sector.

**(A) Revenue model vulnerability — 0 to 40 points**
- 40 = near-zero AI displacement risk (physical assets, regulated monopoly, branded
  physical goods, defence hardware)
- 32 = low (pharma molecules, hospitals, branded consumer goods, auto OEMs)
- 24 = moderate (banks/NBFCs — data moat partly offsets; engineering/EPC)
- 16 = high (headcount/T&M IT services, BPO-adjacent)
- 8 = very high (pure labour arbitrage, no proprietary IP)

**(B) Adaptability & AI optionality — 0 to 35 points**
- 35 = AI-native or platform-led (proprietary models/data moats)
- 28 = active adaptor (clear AI roadmap, meaningful capex, defensible moat)
- 21 = moderate (some investment, reasonable differentiation)
- 14 = passive (reactive stance, no flagship AI product)
- 7 = laggard (no visible AI strategy, shrinking moat)

**(C) Sector demand tailwind from AI — 0 to 25 points**
- 25 = strong pull (defence AI systems, data-centre power/cabling, electronics
  manufacturing, semiconductor-adjacent specialty chemicals)
- 20 = moderate (healthcare AI diagnostics, fintech, telecom infra)
- 15 = neutral (autos, FMCG, pharma generics, cement)
- 10 = mild headwind (commodity IT services)
- 5 = strong headwind (BPO, or the company's own clients are being disrupted)

```python
def score_ai_exposure(revenue_vuln_0to40, adaptability_0to35, demand_tailwind_0to25):
    return revenue_vuln_0to40 + adaptability_0to35 + demand_tailwind_0to25
```

If truly no basis to judge for a brand-new stock, default to **55** (mild positive —
most physical/regulated Indian businesses skew net-neutral-to-positive rather than
net-negative), but write out the reasoning for each sub-score wherever possible instead
of relying on this default.

---

## 8. Sector exclusion list (hard filter — apply before scoring)

A stock is excluded entirely — never scored, never included — if its sector matches any
of the following:

```python
EXCLUDED_SECTORS = {
    "IT",                # IT Services (legacy outsourcing)
    "Energy",             # Coal Mining (verify per-stock — this tag can be used broadly)
    "Oil & Gas",           # Integrated Oil & Gas / OMCs (upstream + refining/marketing)
    "FMCG",                 # FMCG — includes staples, alcoholic beverages, tobacco
    "Insurance",              # Life & Health Insurance
}

EXCLUDED_TICKERS = {
    # Individually excluded — sector taxonomy above is too coarse to isolate these by name:
    "SBIN",         # PSU Bank (sector "Banks" also contains private banks, which are kept)
    "ADANIPORTS",   # Ports & Marine Shipping
}
```

Also check new stocks against these categories, even though no current matches exist in
the tracked universe: **Airlines, Commodity Textiles, Pulp & Paper, Residential Real
Estate, Apparel & Footwear, Media & Entertainment / Broadcasting.**

**Electric utilities are explicitly eligible** (NTPC, Power Grid, Torrent Power, etc.) —
do not exclude by sector for utilities; they're evaluated on merit like everything else.

**Shipbuilders are distinct from ports/shipping lines** — a defence or commercial
shipbuilder is a manufacturing business, not a port operator, and is not covered by the
Ports & Marine Shipping exclusion.

---

## 9. Stagnant-growth exclusion filter

Applied after sector exclusion, using actual growth data — not company age:

```python
def is_stagnant(sg5y, sg3y, pg5y, pg3y):
    """Flag as stagnant only if BOTH long-run revenue and profit growth are
    structurally weak. A company with modest revenue growth but strong profit
    growth (margin expansion, operating leverage) should NOT be excluded."""
    sg = sg5y if sg5y is not None else sg3y
    pg = pg5y if pg5y is not None else pg3y
    return (sg is not None and sg < 8) and (pg is not None and pg < 10)
```

---

## 10. Full scoring function

```python
def score_stock(ticker: str, data: dict, qualitative: dict, sector_avg_pe: float,
                 sector_avg_ev_ebitda: float) -> dict:
    """
    data: quantitative fields fetched from tapetide (Section 1)
    qualitative: manually/LLM-assessed 0-100 scores for every qualitative criterion
                 listed in Sections 3, 4, 5, 6, 7 (missing keys use per-function defaults)
    """
    is_bank = data.get('sector') == 'Banks'

    ev_ebitda = compute_ev_ebitda(data.get('mcap'), data.get('de'), data.get('bv'),
                                    data.get('issued'), data.get('ebitda'))
    peg = compute_peg(data.get('pe'), data.get('pg5y') or data.get('pg3y'))

    A = {
        'capital_efficiency': score_capital_efficiency(data.get('roe'), data.get('roce')),
        'revenue_earnings_growth': score_revenue_earnings_growth(
            data.get('sg1y'), data.get('sg3y'), data.get('sg5y'),
            data.get('pg1y'), data.get('pg3y'), data.get('pg5y')),
        'debt_quality': score_debt_quality(data.get('de'), is_bank),
        'asset_light': qualitative.get('asset_light', 50),
        'equity_dilution_history': qualitative.get('equity_dilution_history', 50),
        'working_capital_trend': qualitative.get('working_capital_trend', 50),
        'dividend_consistency': qualitative.get('dividend_consistency', 50),
        'customer_concentration': qualitative.get('customer_concentration', 50),
    }
    A_WEIGHTS = {'capital_efficiency':.20,'revenue_earnings_growth':.20,'debt_quality':.15,
                 'asset_light':.12,'equity_dilution_history':.10,'working_capital_trend':.10,
                 'dividend_consistency':.08,'customer_concentration':.05}

    B = {
        'peg_ratio': score_peg(peg),
        'pe_vs_sector': score_pe_vs_sector(data.get('pe'), sector_avg_pe),
        'price_to_book': score_price_to_book(data.get('pb')),
        'ev_ebitda': score_ev_ebitda(ev_ebitda, sector_avg_ev_ebitda),
        'fcf_yield': qualitative.get('fcf_yield', 50),
        'forward_growth_potential': qualitative.get('forward_growth_potential', 50),
    }
    B_WEIGHTS = {'peg_ratio':.25,'pe_vs_sector':.20,'price_to_book':.15,
                 'ev_ebitda':.15,'fcf_yield':.15,'forward_growth_potential':.10}

    C = {
        'economic_moat': qualitative.get('economic_moat', 50),
        'management_quality': score_management_quality_proxy(data.get('roce')),
        'market_leadership': qualitative.get('market_leadership', 50),
        'addressable_market': qualitative.get('addressable_market', 50),
        'sector_growth': qualitative.get('sector_growth', 50),
        'revenue_diversification': qualitative.get('revenue_diversification', 50),
        'biz_model_quality': qualitative.get('biz_model_quality', 50),
    }
    C_WEIGHTS = {'economic_moat':.222,'management_quality':.167,'market_leadership':.111,
                 'addressable_market':.111,'sector_growth':.111,'revenue_diversification':.111,
                 'biz_model_quality':.167}

    D = {
        'corporate_governance': qualitative.get('corporate_governance', 50),
        'promoter_holding': qualitative.get('promoter_holding', 50),
        'institutional_ownership': qualitative.get('institutional_ownership', 50),
        'insider_trading': qualitative.get('insider_trading', 50),
        'longevity_consistency': score_longevity(data.get('ld'), qualitative.get('longevity_prior', 50)),
    }
    D_WEIGHTS = {'corporate_governance':.25,'promoter_holding':.25,'institutional_ownership':.20,
                 'insider_trading':.15,'longevity_consistency':.15}

    ai_exposure = score_ai_exposure(
        qualitative.get('ai_revenue_vuln', 24),
        qualitative.get('ai_adaptability', 21),
        qualitative.get('ai_demand_tailwind', 15))

    F = {
        'sector_tailwinds_headwinds': qualitative.get('sector_tailwinds_headwinds', 50),
        'momentum': score_momentum(data.get('ltp'), data.get('h52')),
        'earnings_estimate_revisions': score_analyst_revisions(data.get('buy')),
        'active_macro_event': qualitative.get('active_macro_event', 50),
        'ai_automation_net_exposure': ai_exposure,
    }
    F_WEIGHTS = {'sector_tailwinds_headwinds':.20,'momentum':.24,
                 'earnings_estimate_revisions':.20,'active_macro_event':.16,
                 'ai_automation_net_exposure':.20}

    catA = category_score(A, A_WEIGHTS)
    catB = category_score(B, B_WEIGHTS)
    catC = category_score(C, C_WEIGHTS)
    catD = category_score(D, D_WEIGHTS)
    catF = category_score(F, F_WEIGHTS)

    total = final_score(catA, catB, catC, catD, catF)

    stagnant = is_stagnant(data.get('sg5y'), data.get('sg3y'), data.get('pg5y'), data.get('pg3y'))
    sector_excluded = data.get('sector') in EXCLUDED_SECTORS or ticker in EXCLUDED_TICKERS

    return {
        'ticker': ticker,
        'sector': data.get('sector'),
        'total_score': total,
        'cat_A_financial': round(catA, 2),
        'cat_B_valuation': round(catB, 2),
        'cat_C_fundamentals': round(catC, 2),
        'cat_D_governance': round(catD, 2),
        'cat_F_momentum_ai': round(catF, 2),
        'ai_exposure_score': ai_exposure,
        'ltp': data.get('ltp'),
        'flags': {
            'sector_excluded': sector_excluded,
            'stagnant_growth': stagnant,
        },
        'include_in_list': (total >= 60.0) and (not sector_excluded) and (not stagnant),
    }
```

---

## 11. Operational instructions

1. **New stock request flow:** call `tapetide.search_stocks` to resolve the NSE symbol
   and sector if not already known, then `tapetide.get_company_profile(include=["ratings"])`
   for quantitative data.
2. **Populate qualitative fields with real research**, not flat defaults. Actively look
   into moat, management quality, sector tailwinds, and especially the three
   AI-exposure sub-scores. Write out the reasoning per qualitative score in the output
   so it can be audited.
3. **Apply exclusion filters before presenting a score.** If a stock is sector-excluded
   or stagnant, say so plainly and exclude it from the ranked list even if the raw score
   would have cleared 60.
4. **Threshold rule:** only stocks scoring **≥ 60.0** (and passing both filters) are
   added to the ranked list. Still show the score and category breakdown for stocks that
   don't qualify, so it's clear how close they came.
5. **Master list maintenance:** maintain a single running JSON file (e.g.
   `ranked_stock_list.json`) with schema: rank, ticker, name, sector, total,
   cat_A_financial, cat_B_valuation, cat_C_fundamentals, cat_D_governance,
   cat_F_momentum_ai, ai_exposure_score, ltp. When a new stock qualifies, insert it at
   the correct rank and re-rank the full list — don't just append.
6. **Never fabricate quantitative data.** If tapetide returns null for a required field,
   propagate null and let the scoring functions' built-in defaults handle it.
7. **Flag data-quality issues explicitly** in the output (e.g., `"governance_data":
   "estimated — shareholding API not available"`) rather than presenting an estimated
   score as if it were live data.
8. **Scope note:** this specification covers only the fundamental scoring gate. It does
   not perform sector-level macro analysis or technical entry/exit timing — those are
   separate stages of the broader investment workflow, to be specified independently.
