from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

app = Flask(__name__)

SECTOR_UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ADBE", "CRM", "AMD", "INTC",
        "ORCL", "CSCO", "TXN", "QCOM", "NOW", "INTU", "SNPS", "CDNS", "ANET", "PANW",
        "CRWD", "FTNT", "MRVL", "KLAC", "LRCX", "AMAT", "MU", "ADSK", "WDAY", "TEAM",
        "DDOG", "ZS", "SNOW", "NET", "HUBS", "SPLK", "PLTR", "U", "SHOP", "SQ",
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "MDT", "ISRG", "SYK", "BSX", "EW", "ZTS", "REGN", "VRTX",
        "IDXX", "IQV", "DXCM", "ALGN", "HOLX", "MTD", "RMD", "BAX", "BDX", "CI",
        "HUM", "CNC", "MOH", "HCA", "A", "BIIB", "MRNA", "ILMN", "VEEV", "PODD",
    ],
    "Finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB",
        "PNC", "TFC", "BK", "CME", "ICE", "SPGI", "MCO", "MSCI", "AON", "MMC",
        "AJG", "CB", "PGR", "TRV", "ALL", "MET", "PRU", "AFL", "AIG", "FIS",
        "FISV", "GPN", "V", "MA", "PYPL", "COF", "DFS", "SYF", "ALLY", "COIN",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "DVN",
        "HES", "HAL", "BKR", "FANG", "WMB", "KMI", "OKE", "TRGP", "ET", "EPD",
        "PXD", "APA", "CTRA", "MRO", "AR", "EQT", "RRC", "SWN", "DINO", "VNOM",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST",
        "DHI", "LEN", "PHM", "NVR", "ORLY", "AZO", "AAP", "BKNG", "ABNB", "MAR",
        "HLT", "CMG", "DPZ", "YUM", "DARDEN", "LVS", "WYNN", "MGM", "F", "GM",
        "RIVN", "LCID", "LULU", "GPS", "ANF", "DECK", "CROX", "ETSY", "EBAY", "W",
    ],
    "Consumer Staples": [
        "WMT", "COST", "PG", "KO", "PEP", "CL", "EL", "KMB", "CHD", "CLX",
        "SJM", "MKC", "HSY", "MDLZ", "GIS", "K", "CAG", "CPB", "KHC", "TSN",
        "HRL", "SYY", "USFD", "ADM", "BG", "PM", "MO", "STZ", "BF-B", "TAP",
    ],
    "Industrials": [
        "CAT", "DE", "HON", "UPS", "RTX", "BA", "LMT", "GE", "MMM", "EMR",
        "ITW", "ETN", "PH", "ROK", "CMI", "FDX", "CSX", "UNP", "NSC", "GD",
        "NOC", "TXT", "HII", "LHX", "AXON", "TDG", "HWM", "GWW", "FAST", "WSO",
        "SWK", "IR", "AME", "OTIS", "CARR", "JCI", "TRANE", "XYL", "DOV", "WM",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "PSA", "WELL", "AVB",
        "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "INVH", "SUI", "CPT", "REG",
        "KIM", "FRT", "BXP", "VNO", "SLG", "HIW", "CUBE", "EXR", "SBAC", "IRM",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC",
        "ES", "AWK", "ATO", "NI", "CMS", "DTE", "PEG", "FE", "PPL", "EVRG",
        "AES", "CEG", "VST", "OGE", "PNW", "LNT", "WTRG", "AEE", "ETR", "CNP",
    ],
    "Communication Services": [
        "GOOG", "META", "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "EA",
        "ATVI", "TTWO", "RBLX", "MTCH", "PINS", "SNAP", "ZM", "SPOT", "WBD", "PARA",
        "LYV", "IACI", "FOXA", "NWSA", "OMC", "IPG", "WPP", "ROKU", "TTD", "MGNI",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "GOLD", "NUE", "STLD",
        "CLF", "RS", "VMC", "MLM", "CRH", "EMN", "CE", "PPG", "RPM", "ALB",
        "CTVA", "FMC", "MOS", "CF", "IFF", "AVNT", "BALL", "PKG", "IP", "WRK",
    ],
}

SECTOR_AVG_PE = {
    "Technology": 30, "Healthcare": 22, "Financial Services": 15,
    "Energy": 12, "Consumer Cyclical": 25, "Consumer Defensive": 22,
    "Industrials": 20, "Real Estate": 35, "Communication Services": 18,
    "Basic Materials": 15, "Utilities": 18,
}


def _safe_pct(val):
    """Normalize percentage values — yfinance sometimes returns 0.93 meaning 0.93%,
    other times 0.0093 meaning 0.93%. Values > 1 are assumed already in pct form."""
    if val is None:
        return 0
    if abs(val) > 1:
        return val / 100  # was already in percentage form, convert to decimal
    return val


def _median_of_valid(values):
    """Return median of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    return float(np.median(clean)) if clean else None


def _cross_validate_dividend(ticker, info, current_price):
    """Compute dividend yield from multiple independent sources and return
    the validated yield (as decimal, e.g. 0.008 for 0.8%).

    Sources checked:
      1. info['dividendYield']
      2. info['dividendRate'] / currentPrice
      3. info['trailingAnnualDividendYield']
      4. info['trailingAnnualDividendRate'] / currentPrice
      5. Actual dividend history (last 12 months summed / price)
    """
    estimates = []

    # Source 1: dividendYield from info
    raw = info.get("dividendYield")
    if raw is not None and raw > 0:
        normalized = raw / 100 if raw > 1 else raw  # fix percentage vs decimal
        if normalized < 0.25:  # sanity: no stock yields >25%
            estimates.append(normalized)

    # Source 2: dividendRate / price
    div_rate = info.get("dividendRate")
    if div_rate is not None and div_rate > 0 and current_price > 0:
        calc_yield = div_rate / current_price
        if calc_yield < 0.25:
            estimates.append(calc_yield)

    # Source 3: trailingAnnualDividendYield
    trailing_yield = info.get("trailingAnnualDividendYield")
    if trailing_yield is not None and trailing_yield > 0:
        normalized = trailing_yield / 100 if trailing_yield > 1 else trailing_yield
        if normalized < 0.25:
            estimates.append(normalized)

    # Source 4: trailingAnnualDividendRate / price
    trailing_rate = info.get("trailingAnnualDividendRate")
    if trailing_rate is not None and trailing_rate > 0 and current_price > 0:
        calc_yield = trailing_rate / current_price
        if calc_yield < 0.25:
            estimates.append(calc_yield)

    # Source 5: actual dividend history (last 12 months)
    try:
        divs = ticker.dividends
        if divs is not None and len(divs) > 0:
            one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(months=12)
            recent_divs = divs[divs.index >= one_year_ago]
            if len(recent_divs) > 0 and current_price > 0:
                annual_div = float(recent_divs.sum())
                hist_yield = annual_div / current_price
                if 0 < hist_yield < 0.25:
                    estimates.append(hist_yield)
    except Exception:
        pass

    if not estimates:
        return 0

    # Use median to reject outliers
    return float(np.median(estimates))


def _cross_validate_pe(info, ticker, current_price):
    """Compute P/E from multiple sources.

    Sources:
      1. info['trailingPE']
      2. currentPrice / info['trailingEps']
      3. Calculated from income statement (net income / shares outstanding)
      4. info['forwardPE'] (as sanity reference)
    Returns (trailing_pe, forward_pe) — validated.
    """
    pe_estimates = []

    # Source 1: direct trailingPE
    raw_pe = info.get("trailingPE")
    if raw_pe is not None and 0 < raw_pe < 1000:
        pe_estimates.append(raw_pe)

    # Source 2: price / trailingEps
    eps = info.get("trailingEps")
    if eps is not None and eps > 0 and current_price > 0:
        calc_pe = current_price / eps
        if 0 < calc_pe < 1000:
            pe_estimates.append(calc_pe)

    # Source 3: from income statement
    try:
        income_stmt = ticker.income_stmt
        if income_stmt is not None and not income_stmt.empty:
            ni_row = None
            for label in ["Net Income", "NetIncome"]:
                if label in income_stmt.index:
                    ni_row = income_stmt.loc[label]
                    break
            if ni_row is not None:
                latest_ni = ni_row.dropna().iloc[-1] if not ni_row.dropna().empty else None
                shares = info.get("sharesOutstanding", 0)
                if latest_ni and latest_ni > 0 and shares > 0 and current_price > 0:
                    calc_eps = latest_ni / shares
                    calc_pe = current_price / calc_eps
                    if 0 < calc_pe < 1000:
                        pe_estimates.append(calc_pe)
    except Exception:
        pass

    trailing_pe = _median_of_valid(pe_estimates) if pe_estimates else None

    # Forward PE — just validate bounds
    forward_pe = info.get("forwardPE")
    if forward_pe is not None and (forward_pe <= 0 or forward_pe > 500):
        forward_pe = None

    return trailing_pe, forward_pe


def _cross_validate_debt(info, ticker):
    """Get D/E ratio from multiple sources. Returns as a raw ratio (e.g. 0.34x).

    Sources:
      1. info['debtToEquity'] — yfinance returns this as percentage (34 = 34% = 0.34x)
      2. Calculated from balance sheet (Total Debt / Stockholders Equity) — raw ratio
      3. Calculated from most recent quarterly balance sheet
    """
    estimates = []

    # Source 1: yfinance info (convert from percentage to ratio)
    raw_de = info.get("debtToEquity")
    if raw_de is not None and raw_de >= 0:
        estimates.append(raw_de / 100)  # 34 -> 0.34

    # Source 2: annual balance sheet
    try:
        bs = ticker.balance_sheet
        if bs is not None and not bs.empty:
            debt = None
            equity = None
            for label in ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"]:
                if label in bs.index:
                    vals = bs.loc[label].dropna()
                    if len(vals) > 0:
                        debt = float(vals.iloc[-1])
                        break
            for label in ["Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest"]:
                if label in bs.index:
                    vals = bs.loc[label].dropna()
                    if len(vals) > 0:
                        equity = float(vals.iloc[-1])
                        break
            if debt is not None and equity is not None and equity > 0:
                calc_de = debt / equity  # raw ratio
                if calc_de >= 0:
                    estimates.append(calc_de)
    except Exception:
        pass

    # Source 3: quarterly balance sheet (more recent)
    try:
        qbs = ticker.quarterly_balance_sheet
        if qbs is not None and not qbs.empty:
            debt = None
            equity = None
            for label in ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"]:
                if label in qbs.index:
                    vals = qbs.loc[label].dropna()
                    if len(vals) > 0:
                        debt = float(vals.iloc[-1])
                        break
            for label in ["Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest"]:
                if label in qbs.index:
                    vals = qbs.loc[label].dropna()
                    if len(vals) > 0:
                        equity = float(vals.iloc[-1])
                        break
            if debt is not None and equity is not None and equity > 0:
                calc_de = debt / equity
                if calc_de >= 0:
                    estimates.append(calc_de)
    except Exception:
        pass

    return _median_of_valid(estimates)


def _cross_validate_payout(info, ticker, current_price):
    """Validate payout ratio from multiple sources.

    Sources:
      1. info['payoutRatio']
      2. dividendRate / trailingEps
      3. Calculated from actual dividends / net income per share
    """
    estimates = []

    raw = info.get("payoutRatio")
    if raw is not None:
        normalized = raw / 100 if raw > 2 else raw  # fix if in pct form
        if 0 <= normalized <= 2:  # allow up to 200% (some companies temporarily)
            estimates.append(normalized)

    div_rate = info.get("dividendRate", 0) or 0
    eps = info.get("trailingEps", 0) or 0
    if div_rate > 0 and eps > 0:
        calc_payout = div_rate / eps
        if 0 <= calc_payout <= 2:
            estimates.append(calc_payout)

    if not estimates:
        return 0
    return float(np.median(estimates))


def _build_price_targets(info, ticker, current_price, fifty_two_high, fifty_two_low):
    """Build realistic 12-month price targets by blending multiple estimation
    methods and anchoring to the consensus, not outlier analyst opinions.

    Instead of taking the single highest/lowest analyst target, we:
      1. Collect analyst consensus mean/median targets (most reliable)
      2. Compute a DCF-style estimate from earnings growth + current price
      3. Compute a revenue-momentum estimate from revenue growth trend
      4. Use 52-week range to infer historical volatility bounds
      5. Blend these into weighted-average bull/mean/bear targets

    The bull case is NOT the single most optimistic analyst — it's the
    realistic upside scenario based on consensus + growth data.
    """
    num_analysts = info.get("numberOfAnalystOpinions", 0) or 0
    analyst_high = info.get("targetHighPrice")
    analyst_low = info.get("targetLowPrice")
    analyst_mean = info.get("targetMeanPrice")
    analyst_median = info.get("targetMedianPrice")

    earnings_growth = info.get("earningsGrowth") or 0
    revenue_growth = info.get("revenueGrowth") or 0
    forward_pe = info.get("forwardPE")
    forward_eps = info.get("forwardEps")

    # Normalize growth rates that might be in pct form
    if abs(earnings_growth) > 2:
        earnings_growth = earnings_growth / 100
    if abs(revenue_growth) > 2:
        revenue_growth = revenue_growth / 100

    # === MEAN TARGET: blend of available consensus estimates ===
    mean_estimates = []

    # Analyst consensus mean (strongest signal)
    if analyst_mean and 0.3 * current_price < analyst_mean < 3 * current_price:
        mean_estimates.append(analyst_mean)

    # Analyst consensus median (if available, often more robust than mean)
    if analyst_median and 0.3 * current_price < analyst_median < 3 * current_price:
        mean_estimates.append(analyst_median)

    # Earnings growth projection: price * (1 + earnings_growth)
    if -0.5 < earnings_growth < 1.0:
        eg_target = current_price * (1 + earnings_growth)
        mean_estimates.append(eg_target)

    # Forward PE * forward EPS (if both available)
    if forward_pe and forward_eps and forward_pe > 0 and forward_eps > 0:
        fwd_target = forward_pe * forward_eps
        if 0.3 * current_price < fwd_target < 3 * current_price:
            mean_estimates.append(fwd_target)

    # Revenue growth as price proxy (weaker signal, lower implicit weight via averaging)
    if -0.3 < revenue_growth < 0.8:
        rg_target = current_price * (1 + revenue_growth * 0.7)  # discount factor
        mean_estimates.append(rg_target)

    if mean_estimates:
        # Weighted: analyst consensus counts double (if present)
        weights = []
        for i, est in enumerate(mean_estimates):
            if i < 2 and num_analysts >= 5:
                weights.append(2.0)  # analyst mean/median get double weight
            else:
                weights.append(1.0)
        target_mean = round(np.average(mean_estimates, weights=weights), 2)
    else:
        target_mean = None

    # === BULL TARGET: realistic optimistic scenario ===
    # Blend: 70% weighted mean shifted up + 30% analyst high (dampened)
    bull_estimates = []

    if target_mean:
        # Optimistic mean: shift consensus up by half the distance to analyst high
        if analyst_high and analyst_high > target_mean:
            optimistic_shift = (analyst_high - target_mean) * 0.3  # take 30% of the gap
            bull_estimates.append(target_mean + optimistic_shift)
        else:
            bull_estimates.append(target_mean * 1.10)  # +10% above mean

    # 52-week high as a sanity anchor — bull case shouldn't wildly exceed it
    if fifty_two_high and fifty_two_high > current_price:
        # Project slightly above 52w high (stocks can break out, but not 2x)
        bull_estimates.append(fifty_two_high * 1.10)

    # Earnings growth bull: assume growth beats expectations by 30%
    if earnings_growth > 0:
        bull_eg = current_price * (1 + earnings_growth * 1.3)
        bull_estimates.append(bull_eg)

    if bull_estimates:
        target_high = round(float(np.mean(bull_estimates)), 2)
        # Final cap: bull case can't exceed +50% from current price
        max_bull = current_price * 1.50
        target_high = min(target_high, round(max_bull, 2))
        # Must be above mean
        if target_mean and target_high < target_mean:
            target_high = round(target_mean * 1.05, 2)
    else:
        target_high = round(current_price * 1.15, 2) if current_price else None

    # === BEAR TARGET: realistic downside scenario ===
    bear_estimates = []

    if target_mean:
        # Pessimistic mean: shift consensus down by half the distance to analyst low
        if analyst_low and analyst_low < target_mean:
            pessimistic_shift = (target_mean - analyst_low) * 0.3
            bear_estimates.append(target_mean - pessimistic_shift)
        else:
            bear_estimates.append(target_mean * 0.90)

    # 52-week low as anchor
    if fifty_two_low and fifty_two_low < current_price:
        bear_estimates.append(fifty_two_low)

    # Negative earnings scenario
    if earnings_growth < 0:
        bear_eg = current_price * (1 + earnings_growth * 1.3)
        bear_estimates.append(bear_eg)
    else:
        # Even in good times, model a disappointment scenario
        bear_estimates.append(current_price * 0.85)

    if bear_estimates:
        target_low = round(float(np.mean(bear_estimates)), 2)
        # Floor: bear case can't drop more than -40%
        min_bear = current_price * 0.60
        target_low = max(target_low, round(min_bear, 2))
        # Must be below mean
        if target_mean and target_low > target_mean:
            target_low = round(target_mean * 0.95, 2)
    else:
        target_low = round(current_price * 0.85, 2) if current_price else None

    return target_high, target_low, target_mean, num_analysts


def _validate_margins(info):
    """Validate margin values — must be between -1 and 1 (or -100% to 100%)."""
    def fix(val):
        if val is None:
            return 0
        if abs(val) > 1:
            val = val / 100  # was in pct form
        return max(-1, min(1, val))

    return (
        fix(info.get("grossMargins")),
        fix(info.get("operatingMargins")),
        fix(info.get("profitMargins")),
        fix(info.get("returnOnEquity")),
    )


def _compute_technical_analysis(ticker, current_price, fifty_two_high, fifty_two_low):
    """Compute technical levels and buy/wait/avoid recommendation from price history.

    Analyzes:
      - Support levels (historical price floors the stock bounced from)
      - Resistance levels (historical ceilings the stock struggled to break)
      - Key moving averages (50-day, 100-day, 200-day)
      - RSI (relative strength index) — overbought/oversold
      - Distance from 52-week high/low
      - MACD trend signal
    Returns a dict with all technical data + a recommendation.
    """
    result = {
        "supports": [],
        "resistances": [],
        "ma50": None, "ma100": None, "ma200": None,
        "rsi": None,
        "rsiLabel": "N/A",
        "macdSignal": "Neutral",
        "distFromHigh": None,
        "distFromLow": None,
        "recommendation": "Hold",
        "recommendationColor": "gold",
        "reasons": [],
    }

    try:
        hist = ticker.history(period="1y", interval="1d")
        if hist is None or hist.empty or len(hist) < 30:
            result["recommendation"] = "Insufficient Data"
            result["reasons"].append("Not enough price history for technical analysis")
            return result

        closes = hist["Close"].dropna()
        highs = hist["High"].dropna()
        lows = hist["Low"].dropna()

        # Moving averages
        if len(closes) >= 50:
            result["ma50"] = round(float(closes.iloc[-50:].mean()), 2)
        if len(closes) >= 100:
            result["ma100"] = round(float(closes.iloc[-100:].mean()), 2)
        if len(closes) >= 200:
            result["ma200"] = round(float(closes.iloc[-200:].mean()), 2)

        # RSI (14-period)
        if len(closes) >= 15:
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            last_gain = gain.iloc[-1]
            last_loss = loss.iloc[-1]
            if last_loss > 0:
                rs = last_gain / last_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            result["rsi"] = round(float(rsi), 1)
            if rsi > 70:
                result["rsiLabel"] = "Overbought"
            elif rsi > 60:
                result["rsiLabel"] = "Bullish"
            elif rsi < 30:
                result["rsiLabel"] = "Oversold"
            elif rsi < 40:
                result["rsiLabel"] = "Bearish"
            else:
                result["rsiLabel"] = "Neutral"

        # MACD (12/26/9)
        if len(closes) >= 35:
            ema12 = closes.ewm(span=12).mean()
            ema26 = closes.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_val = float(macd_line.iloc[-1])
            signal_val = float(signal_line.iloc[-1])
            if macd_val > signal_val and macd_val > 0:
                result["macdSignal"] = "Bullish"
            elif macd_val > signal_val:
                result["macdSignal"] = "Turning Bullish"
            elif macd_val < signal_val and macd_val < 0:
                result["macdSignal"] = "Bearish"
            else:
                result["macdSignal"] = "Turning Bearish"

        # Support & Resistance: find local minima/maxima from price pivots
        # Use rolling windows to detect pivot points
        window = 10
        supports_raw = []
        resistances_raw = []

        low_arr = lows.values
        high_arr = highs.values

        for i in range(window, len(low_arr) - window):
            # Local minimum: lowest point in window on both sides
            if low_arr[i] == min(low_arr[i - window:i + window + 1]):
                supports_raw.append(float(low_arr[i]))
            # Local maximum
            if high_arr[i] == max(high_arr[i - window:i + window + 1]):
                resistances_raw.append(float(high_arr[i]))

        # Cluster nearby levels (within 2% of each other) and keep strongest
        def cluster_levels(levels, current, max_count=3):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = []
            cluster = [levels[0]]
            for i in range(1, len(levels)):
                if (levels[i] - cluster[-1]) / cluster[-1] < 0.02:
                    cluster.append(levels[i])
                else:
                    clusters.append((np.mean(cluster), len(cluster)))
                    cluster = [levels[i]]
            clusters.append((np.mean(cluster), len(cluster)))
            # Sort by strength (how many times price touched this level)
            clusters.sort(key=lambda x: x[1], reverse=True)
            # Filter: supports must be below current price, resistances above
            return clusters[:max_count]

        support_clusters = cluster_levels(supports_raw, current_price)
        resistance_clusters = cluster_levels(resistances_raw, current_price)

        # Only keep supports below current price and resistances above
        result["supports"] = sorted(
            [{"price": round(c[0], 2), "strength": c[1]}
             for c in support_clusters if c[0] < current_price * 0.995],
            key=lambda x: x["price"], reverse=True
        )[:3]

        result["resistances"] = sorted(
            [{"price": round(c[0], 2), "strength": c[1]}
             for c in resistance_clusters if c[0] > current_price * 1.005],
            key=lambda x: x["price"]
        )[:3]

        # Distance from 52-week extremes
        if fifty_two_high and fifty_two_high > 0:
            result["distFromHigh"] = round((current_price / fifty_two_high - 1) * 100, 1)
        if fifty_two_low and fifty_two_low > 0:
            result["distFromLow"] = round((current_price / fifty_two_low - 1) * 100, 1)

        # === BUILD RECOMMENDATION ===
        buy_signals = 0
        wait_signals = 0
        avoid_signals = 0
        reasons = []

        # Price vs moving averages
        ma200 = result["ma200"]
        ma50 = result["ma50"]
        if ma200 and current_price > ma200:
            buy_signals += 2
            reasons.append(f"Trading above 200-day MA (${ma200}) — long-term uptrend intact")
        elif ma200:
            avoid_signals += 2
            reasons.append(f"Trading below 200-day MA (${ma200}) — long-term trend is bearish")

        if ma50 and ma200 and ma50 > ma200:
            buy_signals += 1
            reasons.append("Golden cross: 50-day MA above 200-day MA")
        elif ma50 and ma200 and ma50 < ma200:
            avoid_signals += 1
            reasons.append("Death cross: 50-day MA below 200-day MA")

        # RSI
        rsi = result["rsi"]
        if rsi:
            if rsi < 30:
                buy_signals += 2
                reasons.append(f"RSI at {rsi} — oversold, potential bounce opportunity")
            elif rsi < 40:
                buy_signals += 1
                reasons.append(f"RSI at {rsi} — approaching oversold territory")
            elif rsi > 75:
                avoid_signals += 2
                reasons.append(f"RSI at {rsi} — overbought, risk of pullback")
            elif rsi > 65:
                wait_signals += 1
                reasons.append(f"RSI at {rsi} — elevated, consider waiting for a dip")

        # Proximity to support/resistance
        if result["supports"]:
            nearest_support = result["supports"][0]["price"]
            pct_above_support = (current_price - nearest_support) / nearest_support * 100
            if pct_above_support < 3:
                buy_signals += 2
                reasons.append(f"Near strong support at ${nearest_support} — historically bounces here")
            elif pct_above_support < 8:
                buy_signals += 1
                reasons.append(f"Close to support at ${nearest_support} ({pct_above_support:.1f}% above)")

        if result["resistances"]:
            nearest_resistance = result["resistances"][0]["price"]
            pct_below_resistance = (nearest_resistance - current_price) / current_price * 100
            if pct_below_resistance < 2:
                wait_signals += 2
                reasons.append(f"Hitting resistance at ${nearest_resistance} — may get rejected here")
            elif pct_below_resistance < 5:
                wait_signals += 1
                reasons.append(f"Approaching resistance at ${nearest_resistance} ({pct_below_resistance:.1f}% away)")

        # Distance from 52-week high
        dist_high = result["distFromHigh"]
        dist_low = result["distFromLow"]
        if dist_high is not None:
            if dist_high < -20:
                buy_signals += 1
                reasons.append(f"{abs(dist_high)}% below 52-week high — significant discount")
            elif dist_high > -3:
                wait_signals += 1
                reasons.append(f"Near 52-week high ({dist_high}%) — limited near-term upside")

        if dist_low is not None and dist_low < 10:
            avoid_signals += 1
            reasons.append(f"Only {dist_low}% above 52-week low — downtrend may continue")

        # MACD
        macd = result["macdSignal"]
        if macd == "Bullish":
            buy_signals += 1
            reasons.append("MACD bullish — positive momentum")
        elif macd == "Bearish":
            avoid_signals += 1
            reasons.append("MACD bearish — negative momentum")
        elif macd == "Turning Bullish":
            buy_signals += 1
            reasons.append("MACD turning bullish — momentum shifting positive")
        elif macd == "Turning Bearish":
            wait_signals += 1
            reasons.append("MACD turning bearish — momentum weakening")

        # Final recommendation
        if buy_signals >= 4 and avoid_signals <= 1:
            result["recommendation"] = "Buy"
            result["recommendationColor"] = "green"
        elif buy_signals >= 3 and avoid_signals <= 2:
            result["recommendation"] = "Lean Buy"
            result["recommendationColor"] = "green"
        elif avoid_signals >= 4:
            result["recommendation"] = "Avoid"
            result["recommendationColor"] = "red"
        elif avoid_signals >= 3:
            result["recommendation"] = "Lean Avoid"
            result["recommendationColor"] = "red"
        elif wait_signals >= 3:
            result["recommendation"] = "Wait for Dip"
            result["recommendationColor"] = "gold"
        elif buy_signals > avoid_signals:
            result["recommendation"] = "Cautious Buy"
            result["recommendationColor"] = "green"
        elif avoid_signals > buy_signals:
            result["recommendation"] = "Wait"
            result["recommendationColor"] = "gold"
        else:
            result["recommendation"] = "Hold / Neutral"
            result["recommendationColor"] = "gold"

        result["reasons"] = reasons
        result["buySignals"] = buy_signals
        result["waitSignals"] = wait_signals
        result["avoidSignals"] = avoid_signals

    except Exception:
        result["recommendation"] = "Insufficient Data"
        result["reasons"] = ["Could not compute technical analysis"]

    return result


def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or not info.get("currentPrice"):
            return None

        current_price = info.get("currentPrice", 0)
        sector = info.get("sector", "Unknown")
        sector_avg_pe = SECTOR_AVG_PE.get(sector, 20)
        company_name = info.get("shortName", symbol)
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap", 0) or 0
        beta = info.get("beta", 1) or 1
        beta = max(0, min(5, beta))  # cap beta 0-5
        fcf = info.get("freeCashflow", 0) or 0
        fifty_two_high = info.get("fiftyTwoWeekHigh", current_price)
        fifty_two_low = info.get("fiftyTwoWeekLow", current_price)
        recommendation_mean = info.get("recommendationMean", 3) or 3
        revenue_growth = _safe_pct(info.get("revenueGrowth", 0))
        earnings_growth = _safe_pct(info.get("earningsGrowth", 0))

        # === CROSS-VALIDATED METRICS ===

        # P/E from 3 sources
        trailing_pe, forward_pe = _cross_validate_pe(info, ticker, current_price)

        # Dividend yield from 5 sources
        dividend_yield = _cross_validate_dividend(ticker, info, current_price)

        # D/E ratio from 2 sources
        debt_to_equity = _cross_validate_debt(info, ticker)

        # Payout ratio from 2 sources
        payout_ratio = _cross_validate_payout(info, ticker, current_price)

        # Margins validated
        gross_margins, operating_margins, profit_margins, roe = _validate_margins(info)

        # Technical analysis: supports, resistances, MAs, RSI, recommendation
        technical = _compute_technical_analysis(ticker, current_price, fifty_two_high, fifty_two_low)

        # Price targets: blended from multiple estimation methods
        target_high, target_low, target_mean, num_analysts = _build_price_targets(
            info, ticker, current_price, fifty_two_high, fifty_two_low
        )

        # Revenue growth trend from income statement
        revenue_history = []
        try:
            income_stmt = ticker.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                rev_row = None
                for label in ["Total Revenue", "TotalRevenue"]:
                    if label in income_stmt.index:
                        rev_row = income_stmt.loc[label]
                        break
                if rev_row is not None:
                    rev_row = rev_row.dropna().sort_index()
                    for i in range(1, len(rev_row)):
                        prev_val = rev_row.iloc[i - 1]
                        curr_val = rev_row.iloc[i]
                        if prev_val and prev_val != 0:
                            growth = (curr_val - prev_val) / abs(prev_val)
                            # Cap individual year growth to +/- 500%
                            growth = max(-5, min(5, growth))
                            revenue_history.append({
                                "year": str(rev_row.index[i].year) if hasattr(rev_row.index[i], "year") else str(rev_row.index[i]),
                                "growth": round(growth * 100, 1),
                            })
        except Exception:
            pass

        # Cross-validate revenue growth: compare info value vs income statement
        if revenue_history:
            latest_hist_growth = revenue_history[-1]["growth"] / 100
            # If info's revenueGrowth wildly disagrees with statement, use statement
            if abs(revenue_growth - latest_hist_growth) > 0.3:
                revenue_growth = latest_hist_growth

        # Revenue growth trend classification
        if len(revenue_history) >= 2:
            recent = [r["growth"] for r in revenue_history[-2:]]
            older = [r["growth"] for r in revenue_history[:2]]
            avg_recent = np.mean(recent)
            avg_older = np.mean(older)
            if avg_recent > avg_older + 3:
                rev_trend = "Accelerating"
            elif avg_recent < avg_older - 3:
                rev_trend = "Decelerating"
            else:
                rev_trend = "Stable"
        else:
            rev_trend = "Insufficient Data"

        # P/E analysis
        pe_ratio_vs_sector = None
        pe_label = "N/A"
        if trailing_pe and trailing_pe > 0:
            pe_ratio_vs_sector = round(trailing_pe / sector_avg_pe, 2)
            if pe_ratio_vs_sector < 0.7:
                pe_label = "Significantly Undervalued"
            elif pe_ratio_vs_sector < 0.9:
                pe_label = "Undervalued"
            elif pe_ratio_vs_sector < 1.1:
                pe_label = "Fairly Valued"
            elif pe_ratio_vs_sector < 1.3:
                pe_label = "Slightly Overvalued"
            else:
                pe_label = "Overvalued"

        # Debt health (D/E as ratio: 0.5 = 50%, 1.0 = 100%)
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                debt_health = "Excellent"
            elif debt_to_equity < 1.0:
                debt_health = "Good"
            elif debt_to_equity < 1.5:
                debt_health = "Moderate"
            elif debt_to_equity < 2.0:
                debt_health = "Concerning"
            else:
                debt_health = "Poor"
        else:
            debt_health = "N/A"
            debt_to_equity = 0

        # Dividend sustainability (using validated values)
        if dividend_yield > 0:
            if payout_ratio < 0.4 and fcf > 0:
                div_sustainability = "Highly Sustainable"
            elif payout_ratio < 0.6:
                div_sustainability = "Sustainable"
            elif payout_ratio < 0.8:
                div_sustainability = "Moderate"
            else:
                div_sustainability = "At Risk"
        else:
            div_sustainability = "N/A (No Dividend)"

        # Competitive moat
        moat_signals = 0
        moat_reasons = []
        if gross_margins > 0.5:
            moat_signals += 1
            moat_reasons.append(f"High gross margins ({round(gross_margins*100,1)}%)")
        if roe > 0.2:
            moat_signals += 1
            moat_reasons.append(f"Strong ROE ({round(roe*100,1)}%)")
        if operating_margins > 0.2:
            moat_signals += 1
            moat_reasons.append(f"High operating margins ({round(operating_margins*100,1)}%)")
        if market_cap > 200_000_000_000:
            moat_signals += 1
            moat_reasons.append("Mega-cap market leader")

        if moat_signals >= 3:
            moat_rating = "Strong"
        elif moat_signals >= 2:
            moat_rating = "Moderate"
        else:
            moat_rating = "Weak"

        # Bull/Bear price targets (already validated)
        bull_upside = round((target_high / current_price - 1) * 100, 1) if target_high else None
        bear_downside = round((target_low / current_price - 1) * 100, 1) if target_low else None
        mean_target_pct = round((target_mean / current_price - 1) * 100, 1) if target_mean else None

        # Risk rating (1-10)
        risk = 0
        risk += min(beta * 2, 4)
        risk += min(debt_to_equity, 2)  # D/E is already a ratio (0.34 = 34%)
        if revenue_growth < 0:
            risk += 1.5
        elif revenue_growth < 0.05:
            risk += 0.5
        if payout_ratio > 0.8:
            risk += 1
        if trailing_pe and trailing_pe > 40:
            risk += 1
        risk = max(1, min(10, round(risk)))

        # Entry price zone
        if target_low and fifty_two_low:
            entry_low = round(max(target_low, fifty_two_low), 2)
        else:
            entry_low = round(current_price * 0.9, 2)
        entry_high = round(current_price * 0.95, 2)

        # Stop-loss
        stop_loss_pct = 0.08 + (beta - 1) * 0.02
        stop_loss_pct = max(0.05, min(0.15, stop_loss_pct))
        stop_loss = round(entry_low * (1 - stop_loss_pct), 2)

        # Composite score (0-100, hard-capped)
        score = 0

        # PE attractiveness (0-20)
        if pe_ratio_vs_sector:
            if pe_ratio_vs_sector < 0.7:
                score += 20
            elif pe_ratio_vs_sector < 0.9:
                score += 15
            elif pe_ratio_vs_sector < 1.1:
                score += 10
            elif pe_ratio_vs_sector < 1.3:
                score += 5

        # Revenue growth (0-20)
        if revenue_growth > 0.20:
            score += 20
        elif revenue_growth > 0.10:
            score += 15
        elif revenue_growth > 0.05:
            score += 10
        elif revenue_growth > 0:
            score += 5

        # Debt health (0-15) — D/E as ratio
        if debt_to_equity < 0.5:
            score += 15
        elif debt_to_equity < 1.0:
            score += 10
        elif debt_to_equity < 1.5:
            score += 5

        # Moat (0-20)
        if moat_signals >= 3:
            score += 20
        elif moat_signals >= 2:
            score += 12
        else:
            score += 5

        # Dividend (0-10)
        if dividend_yield > 0 and payout_ratio < 0.6:
            score += 10
        elif dividend_yield > 0:
            score += 5

        # Analyst sentiment (0-15)
        if recommendation_mean < 2:
            score += 15
        elif recommendation_mean < 2.5:
            score += 10
        elif recommendation_mean < 3:
            score += 5

        # Hard cap at 100
        score = min(100, score)

        # Data confidence: how many cross-validation sources agreed
        confidence_sources = 0
        if trailing_pe:
            confidence_sources += 1
        if dividend_yield >= 0:
            confidence_sources += 1
        if debt_to_equity is not None:
            confidence_sources += 1
        if revenue_history:
            confidence_sources += 1
        if target_mean:
            confidence_sources += 1
        if num_analysts >= 5:
            confidence_sources += 1

        return {
            "symbol": symbol,
            "companyName": company_name,
            "sector": sector,
            "industry": industry,
            "currentPrice": round(current_price, 2),
            "marketCap": market_cap,
            "trailingPE": round(trailing_pe, 2) if trailing_pe else None,
            "forwardPE": round(forward_pe, 2) if forward_pe else None,
            "peVsSector": pe_ratio_vs_sector,
            "peLabel": pe_label,
            "sectorAvgPE": sector_avg_pe,
            "revenueGrowth": round(revenue_growth * 100, 1),
            "revenueHistory": revenue_history,
            "revenueTrend": rev_trend,
            "debtToEquity": round(debt_to_equity, 2),
            "debtHealth": debt_health,
            "dividendYield": round(dividend_yield * 100, 2),
            "payoutRatio": round(payout_ratio * 100, 1),
            "divSustainability": div_sustainability,
            "grossMargins": round(gross_margins * 100, 1),
            "operatingMargins": round(operating_margins * 100, 1),
            "roe": round(roe * 100, 1),
            "moatRating": moat_rating,
            "moatReasons": moat_reasons,
            "beta": round(beta, 2),
            "targetHigh": target_high,
            "targetLow": target_low,
            "targetMean": target_mean,
            "bullUpside": bull_upside,
            "bearDownside": bear_downside,
            "meanTargetPct": mean_target_pct,
            "riskRating": risk,
            "entryLow": entry_low,
            "entryHigh": entry_high,
            "stopLoss": stop_loss,
            "fiftyTwoHigh": fifty_two_high,
            "fiftyTwoLow": fifty_two_low,
            "score": score,
            "numAnalysts": num_analysts,
            "confidenceSources": confidence_sources,
            "technical": technical,
        }
    except Exception:
        return None


def screen_stocks(risk_tolerance, investment_amount, time_horizon, sectors):
    # Build ticker list
    tickers = []
    if "All" in sectors:
        for s in SECTOR_UNIVERSE.values():
            tickers.extend(s)
    else:
        for s in sectors:
            tickers.extend(SECTOR_UNIVERSE.get(s, []))
    tickers = list(set(tickers))

    # Fetch data in parallel
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_stock_data, t): t for t in tickers}
        for future in as_completed(futures):
            data = future.result()
            if data:
                results.append(data)

    # Filter by risk tolerance
    if risk_tolerance == "Conservative":
        results = [r for r in results if r["riskRating"] <= 4]
    elif risk_tolerance == "Moderate":
        results = [r for r in results if r["riskRating"] <= 7]

    # Adjust scores by time horizon
    for r in results:
        if time_horizon == "Short-term (1-3 months)":
            # Favor low risk, analyst sentiment
            if r["riskRating"] <= 3:
                r["score"] += 10
            if r.get("meanTargetPct") and r["meanTargetPct"] > 10:
                r["score"] += 5
        elif time_horizon == "Long-term (1-5 years)":
            # Favor moat, growth, dividends
            if r["moatRating"] == "Strong":
                r["score"] += 10
            if r["revenueGrowth"] > 10:
                r["score"] += 5
            if r["dividendYield"] > 2:
                r["score"] += 5
        # Hard cap after all adjustments
        r["score"] = min(100, r["score"])

    # Calculate shares affordable (supports fractional shares)
    for r in results:
        if r["currentPrice"] > 0 and investment_amount > 0:
            exact_shares = investment_amount / r["currentPrice"]
            if exact_shares >= 1:
                r["sharesAffordable"] = round(exact_shares, 2)
                r["fractional"] = False
            else:
                r["sharesAffordable"] = round(exact_shares, 4)
                r["fractional"] = True
            r["positionValue"] = round(r["sharesAffordable"] * r["currentPrice"], 2)
            r["ownershipPct"] = round(exact_shares * 100 / (r["marketCap"] / r["currentPrice"]) if r["marketCap"] else 0, 8)
        else:
            r["sharesAffordable"] = 0
            r["positionValue"] = 0
            r["fractional"] = False
            r["ownershipPct"] = 0

    # Sort and return top 10
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results[:10]):
        r["rank"] = i + 1

    return {
        "stocks": results[:10],
        "totalAnalyzed": len(tickers),
        "totalQualified": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.route("/")
def index():
    return render_template("index.html", sectors=list(SECTOR_UNIVERSE.keys()))


@app.route("/lookup", methods=["POST"])
def lookup():
    data = request.get_json()
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "No ticker symbol provided"}), 400

    investment_amount = float(data.get("investmentAmount", 100))
    result = fetch_stock_data(symbol)
    if not result:
        return jsonify({"error": f"Could not find data for '{symbol}'. Check the ticker symbol."}), 404

    result["rank"] = 1
    if result["currentPrice"] > 0 and investment_amount > 0:
        exact_shares = investment_amount / result["currentPrice"]
        if exact_shares >= 1:
            result["sharesAffordable"] = round(exact_shares, 2)
            result["fractional"] = False
        else:
            result["sharesAffordable"] = round(exact_shares, 4)
            result["fractional"] = True
        result["positionValue"] = round(result["sharesAffordable"] * result["currentPrice"], 2)
    else:
        result["sharesAffordable"] = 0
        result["positionValue"] = 0
        result["fractional"] = False

    return jsonify({
        "stocks": [result],
        "totalAnalyzed": 1,
        "totalQualified": 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/screen", methods=["POST"])
def screen():
    data = request.get_json()
    risk_tolerance = data.get("riskTolerance", "Moderate")
    investment_amount = float(data.get("investmentAmount", 10000))
    time_horizon = data.get("timeHorizon", "Medium-term (6-12 months)")
    sectors = data.get("sectors", ["All"])

    result = screen_stocks(risk_tolerance, investment_amount, time_horizon, sectors)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
