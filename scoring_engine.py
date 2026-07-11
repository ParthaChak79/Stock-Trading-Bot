"""
================================================================================
 SCORING ENGINE  —  port of STOCK_SCREENER_SPEC.md
================================================================================
A faithful, deterministic implementation of the fundamental scoring engine
specified in STOCK_SCREENER_SPEC.md. STOCK_SCREENER_SPEC.md is the source of
truth; if the spec changes, update this file in lockstep.

Split of responsibility:
  - Quantitative criteria are computed here from a flat `data` dict (fields
    fetched from TradingView's scanner by tv_fundamentals.py).
  - Qualitative criteria are supplied in a `qualitative` dict (from Gemini via
    gemini_qualitative.py, or per-function spec defaults when unavailable).

Two deviations from the spec's literal code, both to use better live data rather
than the spec's fallback approximations — noted inline:
  1. EV/EBITDA is taken directly from TradingView (`enterprise_value_ebitda_current`)
     instead of reconstructed via compute_ev_ebitda(); the spec's own note prefers
     real values over reconstruction.
  2. fcf_yield is computed from TradingView FCF / market cap (score_fcf_yield)
     instead of defaulted to 50; the spec explicitly says to "calculate
     FCF/market-cap directly where cash flow statement data is available".
================================================================================
"""

CATEGORY_WEIGHTS = {'A': .278, 'B': .142, 'C': .278, 'D': .111, 'F': .191}  # sums to 1.000


def final_score(catA, catB, catC, catD, catF):
    w = CATEGORY_WEIGHTS
    return round(catA * w['A'] + catB * w['B'] + catC * w['C'] + catD * w['D'] + catF * w['F'], 2)


def category_score(criterion_scores: dict, criterion_weights: dict) -> float:
    """Weighted average of available criteria; missing criteria drop out and the
    remaining weights renormalize proportionally."""
    available = {k: v for k, v in criterion_scores.items() if v is not None}
    if not available:
        return 50.0
    total_weight = sum(criterion_weights[k] for k in available)
    return sum(available[k] * criterion_weights[k] / total_weight for k in available)


# ----------------------------------------------------------------------------
# Category A — Financial Quality
# ----------------------------------------------------------------------------
def score_capital_efficiency(roe, roce):
    rs = 50 if roe is None else (100 if roe >= 30 else 90 if roe >= 25 else 80 if roe >= 20
         else 70 if roe >= 15 else 50 if roe >= 10 else 30)
    rcs = 50 if roce is None else (100 if roce >= 25 else 85 if roce >= 20 else 70 if roce >= 15
          else 50 if roce >= 10 else 30)
    return 0.4 * rs + 0.4 * rcs + 10


def score_revenue_earnings_growth(sg1y, sg3y, sg5y, pg1y, pg3y, pg5y):
    rS = 25 if sg1y is None else (50 if sg1y >= 30 else 45 if sg1y >= 20 else 40 if sg1y >= 15
         else 30 if sg1y >= 10 else 20 if sg1y >= 5 else 10 if sg1y > 0 else 0)
    s3 = 20 if sg3y is None else (30 if sg3y >= 25 else 25 if sg3y >= 20 else 20 if sg3y >= 15 else 10)
    s5 = (20 if sg3y and sg3y >= 20 else 15 if sg3y and sg3y >= 15 else 15) if sg5y is None \
         else (20 if sg5y >= 20 else 15 if sg5y >= 15 else 10)
    eR = 25 if pg1y is None else (50 if pg1y >= 30 else 45 if pg1y >= 20 else 40 if pg1y >= 15
         else 30 if pg1y >= 10 else 20 if pg1y >= 5 else 10 if pg1y > 0 else 0)
    e3 = 20 if pg3y is None else (30 if pg3y >= 25 else 25 if pg3y >= 20 else 20 if pg3y >= 15 else 10)
    fg = pg5y if pg5y is not None else pg3y
    e5 = 15 if fg is None else (20 if fg >= 20 else 15 if fg >= 15 else 10)
    return min(100, max(0, 0.6 * (rS + s3 + s5) + 0.4 * (eR + e3 + e5)))


def score_debt_quality(de, is_bank=False):
    if is_bank or de is None:
        return 50  # banks excluded from D/E scoring
    d_score = 50 if de < 0.3 else 45 if de < 0.5 else 35 if de < 1.0 else 20 if de < 2.0 else 5
    interest_proxy = 50 if de < 0.3 else 40 if de < 0.5 else 30 if de < 1.0 else 20 if de < 2.0 else 10
    return d_score + interest_proxy


# ----------------------------------------------------------------------------
# Category B — Valuation
# ----------------------------------------------------------------------------
def score_peg(peg):
    if peg is None:
        return 50
    if peg < 0 or peg > 10:
        return 0
    return (100 if peg < 0.2 else 95 if peg < 0.5 else 85 if peg < 0.8 else 75 if peg < 1.0
            else 60 if peg < 1.5 else 40 if peg < 2.0 else 20)


def score_pe_vs_sector(pe, sector_avg_pe):
    if pe is None or pe <= 0:
        return 0 if pe is not None else 50
    baseline = sector_avg_pe or pe
    discount = (baseline - pe) / baseline * 100
    return (100 if discount >= 50 else 90 if discount >= 30 else 80 if discount >= 20
            else 70 if discount >= 10 else 60 if discount >= 0 else 50 if discount >= -10
            else 35 if discount >= -20 else 20)


def score_price_to_book(pb):
    if pb is None:
        return 50
    if pb < 0:
        return 0
    return 100 if pb < 1 else 85 if pb < 2 else 65 if pb < 3 else 45 if pb < 5 else 25


def score_ev_ebitda(ev_ebitda, sector_avg_ev_ebitda):
    if ev_ebitda is None:
        return 50
    if not sector_avg_ev_ebitda:
        return 55
    discount = (sector_avg_ev_ebitda - ev_ebitda) / sector_avg_ev_ebitda * 100
    return (100 if discount >= 40 else 85 if discount >= 25 else 70 if discount >= 10
            else 55 if discount >= 0 else 40 if discount >= -15 else 20)


def score_fcf_yield(fcf_yield_pct):
    """Deviation-2 helper: score FCF/market-cap directly (spec keeps this criterion
    qualitative-default-50 but instructs to compute it where cash-flow data exists)."""
    if fcf_yield_pct is None:
        return 50
    return (100 if fcf_yield_pct >= 8 else 85 if fcf_yield_pct >= 5 else 70 if fcf_yield_pct >= 3
            else 55 if fcf_yield_pct >= 1 else 45 if fcf_yield_pct > 0 else 25)


def compute_peg(pe, profit_growth_5y_or_3y):
    g = profit_growth_5y_or_3y
    if not pe or pe <= 0 or not g or g <= 0:
        return None
    return pe / min(g, 100)  # cap growth input to avoid absurd PEGs


# ----------------------------------------------------------------------------
# Category C — Business Fundamentals
# ----------------------------------------------------------------------------
def score_management_quality_proxy(roce):
    roce_component = 35 if roce is None else (50 if roce >= 20 else 35 if roce >= 15 else 15)
    return roce_component + 35  # floor 35, max 85


# ----------------------------------------------------------------------------
# Category D — Governance & Ownership
# ----------------------------------------------------------------------------
def score_longevity(listing_date_str, qualitative_prior=50):
    if not listing_date_str:
        return qualitative_prior
    from datetime import datetime
    try:
        age_years = (datetime.now() - datetime.strptime(listing_date_str, '%Y-%m-%d')).days / 365.25
    except Exception:
        return qualitative_prior
    age_score = (50 if age_years >= 25 else 40 if age_years >= 15 else 30 if age_years >= 10
                 else 20 if age_years >= 5 else 10)
    return round(0.4 * age_score + 0.6 * qualitative_prior)


# ----------------------------------------------------------------------------
# Category F — Momentum, Catalysts & AI Exposure
# ----------------------------------------------------------------------------
def score_momentum(ltp, high_52w):
    ratio = ltp / high_52w if (high_52w and ltp) else None
    base = 25 if ratio is None else (50 if ratio >= 0.95 else 35 if ratio >= 0.85
           else 20 if ratio >= 0.70 else 5)
    return base + 20


def score_analyst_revisions(percent_buy):
    if percent_buy is None:
        return 50
    return (100 if percent_buy >= 70 else 85 if percent_buy >= 50 else 65 if percent_buy >= 30
            else 40 if percent_buy >= 15 else 15)


def score_ai_exposure(revenue_vuln_0to40, adaptability_0to35, demand_tailwind_0to25):
    return revenue_vuln_0to40 + adaptability_0to35 + demand_tailwind_0to25


# ----------------------------------------------------------------------------
# Exclusion / stagnation filters
# ----------------------------------------------------------------------------
def is_stagnant(sg5y, sg3y, pg5y, pg3y):
    """Flag as stagnant only if BOTH long-run revenue and profit growth are
    structurally weak."""
    sg = sg5y if sg5y is not None else sg3y
    pg = pg5y if pg5y is not None else pg3y
    return (sg is not None and sg < 8) and (pg is not None and pg < 10)


# ----------------------------------------------------------------------------
# Category weights (per spec)
# ----------------------------------------------------------------------------
A_WEIGHTS = {'capital_efficiency': .20, 'revenue_earnings_growth': .20, 'debt_quality': .15,
             'asset_light': .12, 'equity_dilution_history': .10, 'working_capital_trend': .10,
             'dividend_consistency': .08, 'customer_concentration': .05}
B_WEIGHTS = {'peg_ratio': .25, 'pe_vs_sector': .20, 'price_to_book': .15,
             'ev_ebitda': .15, 'fcf_yield': .15, 'forward_growth_potential': .10}
C_WEIGHTS = {'economic_moat': .222, 'management_quality': .167, 'market_leadership': .111,
             'addressable_market': .111, 'sector_growth': .111, 'revenue_diversification': .111,
             'biz_model_quality': .167}
D_WEIGHTS = {'corporate_governance': .25, 'promoter_holding': .25, 'institutional_ownership': .20,
             'insider_trading': .15, 'longevity_consistency': .15}
F_WEIGHTS = {'sector_tailwinds_headwinds': .20, 'momentum': .24,
             'earnings_estimate_revisions': .20, 'active_macro_event': .16,
             'ai_automation_net_exposure': .20}


def score_stock(ticker: str, data: dict, qualitative: dict,
                sector_avg_pe: float, sector_avg_ev_ebitda: float,
                sector_excluded: bool) -> dict:
    """
    data: quantitative fields from tv_fundamentals.py. Expected keys:
      roe, roce, pe, pb, de, ltp, h52, ev_ebitda, fcf_yield, buy, ld, sector,
      sg1y, sg3y, sg5y, pg1y, pg3y, pg5y, is_bank
    qualitative: 0-100 scores for the qualitative criteria + AI sub-scores
      (ai_revenue_vuln 0-40, ai_adaptability 0-35, ai_demand_tailwind 0-25).
      Missing keys fall back to the spec's per-criterion defaults.
    sector_excluded: precomputed by tv_fundamentals.is_excluded (industry/sector/ticker).
    """
    is_bank = bool(data.get('is_bank'))
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

    B = {
        'peg_ratio': score_peg(peg),
        'pe_vs_sector': score_pe_vs_sector(data.get('pe'), sector_avg_pe),
        'price_to_book': score_price_to_book(data.get('pb')),
        'ev_ebitda': score_ev_ebitda(data.get('ev_ebitda'), sector_avg_ev_ebitda),
        'fcf_yield': score_fcf_yield(data.get('fcf_yield')),
        'forward_growth_potential': qualitative.get('forward_growth_potential', 50),
    }

    C = {
        'economic_moat': qualitative.get('economic_moat', 50),
        'management_quality': score_management_quality_proxy(data.get('roce')),
        'market_leadership': qualitative.get('market_leadership', 50),
        'addressable_market': qualitative.get('addressable_market', 50),
        'sector_growth': qualitative.get('sector_growth', 50),
        'revenue_diversification': qualitative.get('revenue_diversification', 50),
        'biz_model_quality': qualitative.get('biz_model_quality', 50),
    }

    D = {
        'corporate_governance': qualitative.get('corporate_governance', 50),
        'promoter_holding': qualitative.get('promoter_holding', 50),
        'institutional_ownership': qualitative.get('institutional_ownership', 50),
        'insider_trading': qualitative.get('insider_trading', 50),
        'longevity_consistency': score_longevity(data.get('ld'), qualitative.get('longevity_prior', 50)),
    }

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

    catA = category_score(A, A_WEIGHTS)
    catB = category_score(B, B_WEIGHTS)
    catC = category_score(C, C_WEIGHTS)
    catD = category_score(D, D_WEIGHTS)
    catF = category_score(F, F_WEIGHTS)
    total = final_score(catA, catB, catC, catD, catF)

    stagnant = is_stagnant(data.get('sg5y'), data.get('sg3y'), data.get('pg5y'), data.get('pg3y'))

    return {
        'ticker': ticker,
        'name': data.get('name') or ticker,
        'sector': qualitative.get('sector_label') or data.get('sector'),
        'total': total,
        'catA': round(catA, 2),
        'catB': round(catB, 2),
        'catC': round(catC, 2),
        'catD': round(catD, 2),
        'catF': round(catF, 2),
        'ai_score': ai_exposure,
        'ltp': data.get('ltp'),
        'flags': {
            'sector_excluded': sector_excluded,
            'stagnant_growth': stagnant,
        },
        'include_in_list': (total >= 60.0) and (not sector_excluded) and (not stagnant),
    }
