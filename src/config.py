# src/config.py

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. DYNAMIC TIME WINDOWS
# ==========================================
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Minimum history required for the dataset (used for target filtering and data fetching)
MIN_YEARS_HISTORY = 10  
START_DATE = (datetime.today() - timedelta(days=365 * MIN_YEARS_HISTORY)).strftime('%Y-%m-%d')

# ==========================================
# 2. PREDICTION TARGET & ECONOMIC PARAMETERS
# ==========================================
TARGET_ETF = 'ZPRS.DE'
FORECAST_HORIZON_DAYS = 126    # Forecast for 6 months into the future

ANNUAL_INFLATION_RATE = 0.025  # 2.5 % base growth p.a.
ANNUAL_MARGIN_UP = 0.125       # 12.5 % upward tolerance
ANNUAL_MARGIN_DOWN = 0.001      # 0.1 % downward tolerance
TRADING_DAYS_PER_YEAR = 252

# ==========================================
# 3. THE 360-DEGREE INVESTMENT UNIVERSE
# ==========================================

# Macro Basics (Interest rates, Volatility, Currencies)
MACRO_INDICATORS = [
    '^TNX',    # US 10-Year Treasury Yield
    '^IRX',    # 13-Week Treasury Bill 
    '^VIX',    # Volatility Index 
    'DX-Y.NYB',# US Dollar Index
]

# Industrial & Precious Metals
COMMODITIES = [
    'CL=F',    # Crude Oil 
    'GC=F',    # Gold 
    'HG=F',    # Copper ("Dr. Copper")
]

# Agricultural Commodities (Inflation Indicators)
AGRI_COMMODITIES = [
    'ZC=F',    # Corn (Mais)
    'ZW=F',    # Wheat (Weizen)
    'LE=F',    # Live Cattle (Lebendrind)
]

# Credit Risk & Bonds (Leading indicators for stress)
CREDIT_RISK = [
    'HYG',     # High Yield Corporate Bonds (Junk Bonds)
    'TLT',     # 20+ Year Treasury Bonds (Safe Haven)
    'LQD',     # Investment Grade Corporate Bonds
]

# Broad Sectors & International Indices
SECTORS_AND_INDICES = [
    'XLF',     # Financials
    'XLK',     # Technology
    'XLE',     # Energy
    'XBI',     # Biotech
    'EEM',     # Emerging Markets
    '^GDAXI',  # DAX Index
    '^N225',   # Nikkei 225
]

# Sector Rotation (Offensive vs. Defensive)
MORE_SECTORS = [
    'XLU',     # Utilities (Defensive)
    'XLP',     # Consumer Staples (Defensive)
    'XLY',     # Consumer Discretionary (Offensive/Cyclical)
    'XLV',     # Healthcare (Defensive)
]

# Real Estate (Extremely interest rate sensitive)
REAL_ESTATE = [
    'VNQ',     # Vanguard Real Estate ETF
]

# Alternative Liquidity
CRYPTO = [
    'BTC-USD', # Bitcoin (Proxy for global excess liquidity)
]

# Systemically relevant individual stocks
TICKERS_US = ['AAPL', 'MSFT', 'NVDA', 'BRK-B', 'JPM']
TICKERS_DE = ['SAP.DE', 'SIE.DE', 'BAS.DE']
TICKERS_UK = ['SHEL.L', 'AZN.L', 'RIO.L']
TICKERS_JP = ['7203.T', '9984.T', '8035.T']

# Federal Reserve Data (US)
FRED_INDICATORS = [
    'CPIAUCSL',  # US CPI (Inflation)
    'PAYEMS',    # US Nonfarm Payrolls
    'UNRATE',    # US Unemployment
    'T10Y2Y',    # US 10Y-2Y Yield Spread
    'WALCL'      # US Fed Total Assets
]

# FRED: Eurozone & Germany Macro
FRED_INDICATORS_EU = [
    'CP00MI15EA20M086NEST', # Euro Area CPI (Inflation)
    'LRHUTTTTEZM156S',   # Euro Area Unemployment Rate
    'ECBASSETS',         # ECB Total Assets (Systemic Liquidity)
    'PRINTO01EZQ661S'    # Euro Area Industrial Production
]

# FRED: Japan Macro
FRED_INDICATORS_JP = [
    'JPNCPIALLMINMEI',   # Japan CPI
    'LRHUTTTTJPM156S',   # Japan Unemployment Rate
    'JPNASSETS',         # Bank of Japan Total Assets
    'JPNPROINDMISMEI'    # Japan Industrial Production
]

# FRED: UK Macro
FRED_INDICATORS_UK = [
    'GBRCPIALLMINMEI',   # UK CPI
    'LRHUTTTTGBM156S',   # UK Unemployment Rate
    'GBRPROINDMISMEI'    # UK Industrial Production
]

# Yahoo Finance: International Sovereign Bond Yields (ETF Proxies)
SOVEREIGN_YIELDS = [
    'IGOV',  # iShares International Treasury Bond ETF (Heavy exposure to Eurozone/Japan)
    'BWX',   # SPDR Bloomberg International Treasury Bond ETF (Broad non-US sovereign debt)
    'BNDX'   # Vanguard Total International Bond ETF (Hedged non-US bonds)
]

# FRED: Political & Economic Uncertainty
FRED_INDICATORS_UNCERTAINTY = [
    'USEPUINDXD',  # Economic Policy Uncertainty Index for United States
    'GEPUCURRENT', # Global Economic Policy Uncertainty Index
]

# ==========================================
# 4. LEADING INDICATORS & LIQUIDITY (6-12 MONTH HORIZON)
# ==========================================

# Monetary Liquidity & Financial Conditions (FRED)
# Usually impact the real economy and markets with a 6-12 month delay
FRED_LIQUIDITY_AND_CREDIT = [
    'M2SL',           # M2 Money Supply (The true money supply in the system - the fuel for asset prices)
    'NFCI',           # Chicago Fed National Financial Conditions Index (Measures systemic stress & liquidity)
]

# Real Economic Leading Indicators (FRED)
# Often lead the business cycle by 2-3 quarters
FRED_LEADING_MACRO = [
    'PERMIT',         # New Privately-Owned Housing Units Authorized (Building permits lead the economic cycle)
    'ICSA',           # Initial Claims for Unemployment (Weekly initial claims, reacts months before the unemployment rate)
    'UMCSENT',        # Univ. of Michigan: Consumer Sentiment (Consumption accounts for 70% of US GDP)
    'DGORDER',        # Manufacturers' New Orders: Durable Goods (Manufacturers' new orders for durable goods)
]

# Market Cyclicality & Factor ETFs (Yahoo Finance)
# Rotation between these factors often heralds long-term market trends
FACTOR_ETFS = [
    'IWM',            # Russell 2000 ETF (Small Caps - highly dependent on domestic economy, usually turn first)
    'IYT',            # Transportation Average ETF (Dow Theory: Transportation confirms industrial growth)
    'RSP',            # S&P 500 Equal Weight (Shows true market breadth, uninfluenced by a few tech giants)
    'SMH',            # Semiconductor ETF (Semiconductors are the most extreme leading indicator for the global tech cycle)
]

# Specific Leading Indicator Commodities & Currencies (Yahoo Finance)
MORE_COMMODITIES_AND_FX = [
    'LBS=F',          # Random Length Lumber (Lumber - extreme leading indicator for the US housing market)
    'EURUSD=X',       # Euro / US-Dollar
    'JPY=X',          # USD / Japanese Yen (The most important proxy for the global "carry trade" and systemic risks)
]

# Combine all FRED indicators for the pipeline
ALL_FRED_INDICATORS = FRED_INDICATORS + FRED_INDICATORS_EU + FRED_INDICATORS_JP + FRED_INDICATORS_UK + FRED_INDICATORS_UNCERTAINTY + FRED_LIQUIDITY_AND_CREDIT + FRED_LEADING_MACRO

# ==========================================
# MASTER TICKER LIST (BATCH & SINGLE RUN)
# ==========================================
# 'SPY' is hardcoded to prevent data drops when calculating
# systemic interaction ratios (e.g., SPY/TLT).
ALL_TICKERS = list(set(
    ['SPY'] + 
    MACRO_INDICATORS + COMMODITIES + AGRI_COMMODITIES + 
    CREDIT_RISK + SECTORS_AND_INDICES + MORE_SECTORS + 
    REAL_ESTATE + CRYPTO + 
    TICKERS_US + TICKERS_DE + TICKERS_UK + TICKERS_JP +
    SOVEREIGN_YIELDS + FACTOR_ETFS + MORE_COMMODITIES_AND_FX
))

def get_all_tickers():
    """Backwards compatibility function for older pipeline scripts."""
    return ALL_TICKERS

# ==========================================
# 5. LLM CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# gemini-2.5-flash is retired in the Gemini API on 2026-12-01; keep scripts/update_catalog.py in line
GEMINI_MODEL = "gemini-3.5-flash-lite"