#!/usr/bin/env python3
"""
AdvancedGEM.py – Finalny Skrypt DDD + Clean Arch + SOLID + OOP + Wzorce Projektowe

Zawiera:
  - Strategie: TAAStrategy, MomentumTAAStrategy, SmartPriorityTAAStrategy,
    GEMStrategy, SmartGEMStrategy, SmartGEMStrategyTAA, SuperIKEStrategy, IKZEStrategy,
    CryptoStrategy, DCAStrategy, EDOStrategy, SmartGoldenTAA.
  - Mixiny: EffectivenessMixin, SidewaysDetector, SoftStopMixin.
  - Nowe funkcjonalności: HysteresisDecorator, PartialAllocator, BlendStrategy,
    MarketRegimeDetector, SeasonalityFilter, rebalancing, FinalStrategyWrapper.
  - Warstwa Application: CQRS, ReportBuilder, StrategyRunner, EventBus, MultiStrategyPortfolio itd.

Zastosowano defensywne rzutowanie, limitowanie alokacji (min. 5%, max. 85%) oraz wzorce:
  - Strategia, Fabryka, Fasada, CQRS & Event Bus.

Autor: Twoje Imię, data: 2025-04-12
"""

# ====================================================
# IMPORTY & KONFIGURACJA
# ====================================================
import yfinance as yf
import pandas as pd
import warnings
import logging
import numpy as np
from datetime import datetime
from pandas_datareader import data as pdr
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("yfinance").setLevel(logging.ERROR)

# Konfiguracja dat i alokacji
START_DATE = "2023-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

TAA_ALLOCATIONS = {
    "Przewartościowany": {"Akcje": 6,   "Obligacje": 32, "Złoto": 32, "Cash": 30},
    "Bardzo Wysoki":     {"Akcje": 20,  "Obligacje": 28, "Złoto": 28, "Cash": 24},
    "Średnio-Wysoki":    {"Akcje": 40,  "Obligacje": 20, "Złoto": 20, "Cash": 20},
    "Średni":            {"Akcje": 60,  "Obligacje": 15, "Złoto": 15, "Cash": 10},
    "Średnio-Niski":     {"Akcje": 80,  "Obligacje": 7,  "Złoto": 7,  "Cash": 6},
    "Bardzo Niski":      {"Akcje": 100, "Obligacje": 0,  "Złoto": 0,  "Cash": 0}
}

ALT_MAPPING_TAA = {
    "Bardzo Niski":      ("2-ETF: 50% VWCE / 50% NTSG", "1-ETF: VWCE"),
    "Średnio-Niski":     ("3-ETF: EUNA 12%  VWCE 48% / 40% NTSG", "1-ETF: V80A"),
    "Średni":            ("3-ETF: EUNA 28%  VWCE 42% / 30% NTSG", "1-ETF: V60A"),
    "Średnio-Wysoki":    ("3-ETF: EUNA 48%  VWCE 32% / 20% NTSG", "1-ETF: V40A"),
    "Bardzo Wysoki":     ("2-ETF: EUNA 72%  VWCE 18% / 10% NTSG", "1-ETF: V20A"),
    "Przewartościowany": ("Wstrzymaj się (pauza) – 100% cash / ROD", "-")
}

GEM_ASSETS = {
    "S&P 500": "SPY",
    "ACWI ex-US": "ACWX",
    "Emerging": "EEM",
    "Obligacje": "TLT",
    "Gold": "GLD",
    "Cash": "BIL"
}

GEM_DCA_ASSETS = {
    "VWCE": "VWCE",
    "Obligacje": "EUNA"
}

GEM_MOMENTUM = {
    "Momentum Factor": "IWMO.UK",    
    "Obligacje": "TLT",
    "Gold": "GLD",
    "Cash": "BIL"
}

CRYPTO_GEM_ETFS = {"Crypto": "VBTC.DE", "Cash": "BIL"}
IKZE_GEM_ETFS = {"USA (SPY)": "SPY", "ex-US (ACWX)": "ACWX", "Emerging": "EEM", "Gold (GLD)": "GLD", "Obligacje": "TLT", "Cash (BIL)": "BIL"}

SuperIKE_GEM_ETFS = {
    "USA (SPY)": "SPY",
    "Gold": "GLD",
    "Beta CASH": "ETFBCASH.WA",
    "TBSP": "ETFBTBSP.WA",
    "Amundi DAX (ACWX)": "ETFDAX.WA",
    "MSCI Poland": "EPOL"
}

# ====================================================
# INFRASTRUCTURE – DataFetcher oraz MomentumCalculator
# ====================================================
class DataFetcher:
    """Pobiera dane z Yahoo i FRED z wykorzystaniem cache i defensywnego sprawdzania."""
    _cache = {}

    @classmethod
    def fetch_yf(cls, ticker: str) -> pd.Series:
        if ticker in cls._cache:
            return cls._cache[ticker]
        try:
            data = yf.download(ticker, start=START_DATE, end=TODAY, progress=False)["Close"].dropna()
            cls._cache[ticker] = data
            return data
        except Exception as e:
            logging.error(f"Błąd pobierania danych dla {ticker}: {e}")
            return pd.Series(dtype=float)

    @classmethod
    def fetch_fred(cls, symbol: str):
        if symbol in cls._cache:
            return cls._cache[symbol]
        try:
            df = pdr.DataReader(symbol, "fred")
            val = df.dropna().iloc[-1].item()
        except Exception:
            val = None
        cls._cache[symbol] = val
        return val

class MomentumCalculator:
    """Kalkulator momentum dla strategii TAA i GEM z jawnie rzutowanymi wartościami."""
    @staticmethod
    def taa_momentum(data: pd.Series) -> float:
        if len(data) < 252:
            return 0.0
        try:
            roc_1   = float(data.pct_change(1).iloc[-1])
            roc_5   = float(data.pct_change(5).iloc[-1])
            roc_21  = float(data.pct_change(21).iloc[-1])
            roc_63  = float(data.pct_change(63).iloc[-1])
            roc_126 = float(data.pct_change(126).iloc[-1])
            roc_252 = float(data.pct_change(252).iloc[-1])
            sma_200 = float(data.rolling(200).mean().iloc[-1])
            current = float(data.iloc[-1])
            above_sma = 1 if current > sma_200 else 0
            score = 0.1 + roc_1 + 0.1 + roc_5 + 0.2 * roc_21 + 0.2 * roc_63 + 0.2 * roc_126 + 0.1 * roc_252 + 0.1 * above_sma
            vol = float(data.pct_change().std() * (252 ** 0.5))
            return score / vol if vol > 0 else score
        except Exception:
            return 0.0

    @staticmethod
    def gem_momentum(data: pd.Series) -> float:
        if len(data) < 252:
            return 0.0
        try:
            roc_1   = data.pct_change(1).iloc[-1]
            roc_5   = data.pct_change(5).iloc[-1]
            roc_21  = data.pct_change(21).iloc[-1]
            roc_63  = data.pct_change(63).iloc[-1]
            roc_126 = data.pct_change(126).iloc[-1]
            roc_252 = data.pct_change(252).iloc[-1]
            sma_200 = data.rolling(200).mean().iloc[-1]
            current = data.iloc[-1]
            above_sma = 1 if current > sma_200 else 0
            score = 0.1 * roc_1 + 0.1 * roc_5 + 0.2 * roc_21 + 0.2 * roc_63 + 0.2 * roc_126 + 0.1 * roc_252 + 0.1 * above_sma
            vol = data.pct_change().std() * (252 ** 0.5)
            return score / vol if vol > 0 else score
        except Exception:
            return 0.0

# ====================================================
# DOMAIN – Strategie Inwestycyjne
# ====================================================
class InvestmentStrategy(ABC):
    """Abstrakcyjny interfejs strategii inwestycyjnej."""
    @abstractmethod
    def run(self):
        pass

# --- 1) TAAStrategy ---
class TAAStrategy(InvestmentStrategy):
    """Heurystyka TAA: uwzględnia momentum SPY, ryzyko (VIX, FRED) oraz wycenę."""
    def run(self):
        data = DataFetcher.fetch_yf("SPY")
        if data.empty:
            return ("Brak danych", {"Cash": 100}, ("-", "-"))
        mom_val = MomentumCalculator.taa_momentum(data)
        ms = self._calc_momentum_score(mom_val)
        rs = self._calc_risk_score()
        vs = self._calc_valuation_score()
        total = rs + vs + ms
        scenario = self._classify(total, mom_val)
        alt = ALT_MAPPING_TAA.get(scenario, ("-", "-"))
        return (scenario, TAA_ALLOCATIONS.get(scenario, {"Cash": 100}), alt)

    def _calc_momentum_score(self, val: float) -> float:
        score = 6
        for t in [0.00, 0.05, 0.10, 0.15, 0.20]:
            if val > t:
                score -= 1
        if val <= 0:
            score += 2
        return score / 6

    def _calc_risk_score(self) -> float:
        score = 0
        vix = DataFetcher.fetch_yf("^VIX")
        vix_val = float(vix.iloc[-1]) if not vix.empty else 20.0
        for lvl in [18, 22, 26, 30, 34]:
            if vix_val > lvl:
                score += 1
        move_val = DataFetcher.fetch_fred("MOVE")
        if move_val:
            for t in [120, 125, 130, 135, 140]:
                if move_val > t:
                    score += 1
        spread = DataFetcher.fetch_fred("BAMLH0A0HYM2")
        if spread:
            if spread > 4:   score += 1.25
            if spread > 4.5: score += 1.5
            if spread > 5.5: score += 1.25
            if spread > 6:   score += 1
        return score / 5

    def _calc_valuation_score(self) -> float:
        try:
            info = yf.Ticker("SPY").info
            pe = float(info.get("trailingPE", 25.0))
            cape = float(info.get("forwardPE", pe + 5.0))
        except Exception:
            pe, cape = 25.0, 30.0
        sc = sum(pe > x for x in [20, 22, 24, 26, 28]) + sum(cape > x for x in [22, 26, 28, 30, 32])
        return sc / 5

    def _classify(self, score, momentum):
        if score > 2.9:
            return "Przewartościowany"
        elif score > 2.0 and momentum < 0:
            return "Bardzo Wysoki"
        elif score > 2.5:
            return "Bardzo Wysoki"
        elif score > 2.0:
            return "Średnio-Wysoki"
        elif score > 1.4:
            return "Średni"
        elif score > 0.7:
            return "Średnio-Niski"
        return "Bardzo Niski"

# --- 2) MomentumTAAStrategy ---
class MomentumTAAStrategy(InvestmentStrategy):
    """Strategia TAA oparta wyłącznie na momentum dla wybranych aktywów."""
    def __init__(self):
        self.assets = {"ACWI": "Akcje", "TLT": "Obligacje", "GLD": "Złoto", "SHV": "Cash"}
        self.bounds = {"ACWI": (0.05, 1), "TLT": (0.05, 0.55), "GLD": (0.05, 0.85), "SHV": (0.05, 0.55)}
        self.priority = ["ACWI", "TLT", "GLD", "SHV"]

    def run(self):
        scores = {}
        for tkr in self.assets:
            data = DataFetcher.fetch_yf(tkr)
            sc = MomentumCalculator.taa_momentum(data)
            scores[tkr] = max(0, sc)
        sr = pd.Series(scores).sort_values(ascending=False)
        ssum = sr.sum()
        if ssum == 0:
            print("Momentum OFF", {"Cash": 100}, ("-", "-"))
            return ("Momentum TAA", {"Cash": 100}, ("-", "-"))
        raw = {k: v / ssum for k, v in sr.items()}
        bounded = self._apply_max_bounds(raw)
        final = self._redistribute(bounded)
        named = {self.assets[tkr]: round(w * 100, 1) for tkr, w in final.items()}
        return ("Momentum TAA", named, ("-", "-"))

    def _apply_max_bounds(self, raw):
        bounded = {}
        for tkr, w in raw.items():
            _, mx = self.bounds[tkr]
            bounded[tkr] = min(w, mx)
        leftover = 1.0 - sum(bounded.values())
        if leftover > 0:
            for tkr in self.priority:
                _, mx = self.bounds[tkr]
                room = mx - bounded[tkr]
                if room > 0:
                    add = min(room, leftover)
                    bounded[tkr] += add
                    leftover -= add
                if leftover <= 0:
                    break
        return bounded

    def _redistribute(self, weights):
        leftover = 1.0 - sum(weights.values())
        if leftover <= 0.00001:
            return weights
        for tkr in self.priority:
            mn, mx = self.bounds[tkr]
            curr = weights[tkr]
            room = mx - curr
            if room > 0:
                add = min(room, leftover)
                weights[tkr] += add
                leftover -= add
            if leftover <= 0:
                break
        return weights

# --- 3) SmartPriorityTAAStrategy ---
class BlendWeightFactory:
    DEFENSIVE = {"Przewartościowany": 0.05, "Bardzo Wysoki": 0.15, "Średnio-Wysoki": 0.25, "Średni": 0.35, "Średnio-Niski": 0.45, "Bardzo Niski": 0.60}
    BALANCED  = {"Przewartościowany": 0.15, "Bardzo Wysoki": 0.30, "Średnio-Wysoki": 0.45, "Średni": 0.55, "Średnio-Niski": 0.65, "Bardzo Niski": 0.80}
    AGGRESSIVE= {"Przewartościowany": 0.10, "Bardzo Wysoki": 0.35, "Średnio-Wysoki": 0.50, "Średni": 0.65, "Średnio-Niski": 0.75, "Bardzo Niski": 0.90}

    @classmethod
    def get(cls, style: str) -> dict:
        mapping = {"defensive": cls.DEFENSIVE, "balanced": cls.BALANCED, "aggressive": cls.AGGRESSIVE}
        return mapping.get(style.lower(), cls.BALANCED)

class SmartPriorityTAAStrategy(InvestmentStrategy):
    """Strategia TAA łącząca heurystykę i momentum z użyciem blendu wag."""
    def __init__(self, blend_style="balanced"):
        self.blend_weights = BlendWeightFactory.get(blend_style)

    def run(self):
        taa = TAAStrategy()
        scenario, alloc_taa, alt = taa.run()
        mom = MomentumTAAStrategy()
        desc, alloc_mom, _ = mom.run()
        mom_w = self.blend_weights.get(scenario, 0.5)
        taa_w = 1.0 - mom_w
        if mom_w == 0.0:
            return (f"Smart TAA ({scenario}) → Momentum OFF", alloc_taa, alt)
        blended = self._blend_allocations(alloc_taa, alloc_mom, taa_w, mom_w)
        if set(alloc_mom.keys()) == {"Cash"} and alloc_mom.get("Cash", 0) >= 99:
            return (f"Smart TAA ({scenario}) → Momentum OFF", alloc_taa, alt)
        return (f"Smart TAA ({scenario}) – Blend {int(taa_w*100)}/{int(mom_w*100)}", blended, alt)
    
    def _blend_allocations(self, a1, a2, w1, w2):
        keys = set(a1) | set(a2)
        blended = {k: round(a1.get(k, 0) * w1 + a2.get(k, 0) * w2, 1) for k in keys}
        total = sum(blended.values())
        if total > 0:
            blended = {k: round(v / total * 100, 1) for k, v in blended.items()}
        return blended

# --- 4) GEMStrategy ---
class GEMStrategy(InvestmentStrategy):
    """Prosta strategia GEM – wybiera topN aktywów z GEM_ASSETS, resztę alokuje w Cash."""
    def __init__(self, assets=GEM_ASSETS, topn=3, amount=100):
        self.assets = assets
        self.topn = topn
        self.amount = amount

    def run(self):
        scores = {}
        for name, tkr in self.assets.items():
            data = DataFetcher.fetch_yf(tkr)
            if not data.empty:
                scores[name] = MomentumCalculator.gem_momentum(data)
        sr = pd.Series(scores).sort_values(ascending=False)
        top = sr[sr > 0].head(self.topn)
        if top.empty:
            return {"Cash": self.amount}
        ssum = top.sum()
        return {k: round(v / ssum * self.amount, 2) for k, v in top.items()}


class GEMDCAStrategy(InvestmentStrategy):
    """Prosta strategia GEM – wybiera topN aktywów z GEM_ASSETS, resztę alokuje w Cash."""
    def __init__(self, assets=GEM_DCA_ASSETS, topn=3, amount=100):
        self.assets = assets
        self.topn = topn
        self.amount = amount

    def run(self):
        scores = {}
        for name, tkr in self.assets.items():
            data = DataFetcher.fetch_yf(tkr)
            if not data.empty:
                scores[name] = MomentumCalculator.gem_momentum(data)
        sr = pd.Series(scores).sort_values(ascending=False)
        top = sr[sr > 0].head(self.topn)
        if top.empty:
            return {"Cash": self.amount}
        ssum = top.sum()
        return {k: round(v / ssum * self.amount, 2) for k, v in top.items()}


# --- 5) SmartGEMStrategy ---
class SmartGEMStrategy(InvestmentStrategy):
    """Strategia GEM z filtrem ryzyka i override top1."""
    def __init__(self, assets=GEM_ASSETS, topn=3, amount=100, top1_override=0.1):
        self.assets = assets
        self.topn = topn
        self.amount = amount
        self.top1_override = top1_override

    def run(self):
        scores = {}
        for name, tkr in self.assets.items():
            data = DataFetcher.fetch_yf(tkr)
            scores[name] = MomentumCalculator.gem_momentum(data)
        sr = pd.Series(scores).sort_values(ascending=False)
        top = sr[sr > 0].head(self.topn)
        risk = TAAStrategy()._calc_risk_score()
        if risk > 2.5:
            print("🔻 Wysokie ryzyko rynkowe – Cash")
            return {"Cash": self.amount}
        if len(top) >= 2 and (top.iloc[0] - top.iloc[1] > self.top1_override):
            return {top.index[0]: self.amount}
        if top.empty:
            return {"Cash": self.amount}
        ssum = top.sum()
        return {k: round(v / ssum * self.amount, 2) for k, v in top.items()}

# --- 6) SmartGEMStrategyTAA ---
class SmartGEMStrategyTAA(InvestmentStrategy):
    """Strategia GEM z dynamicznym topn określanym wg scenariusza TAA."""
    taa_gem_config = {
        "Przewartościowany": {"enabled": False, "topn": 0, "top1_threshold": None},
        "Bardzo Wysoki": {"enabled": True, "topn": 1, "top1_threshold": 0.15},
        "Średnio-Wysoki": {"enabled": True, "topn": 2, "top1_threshold": 0.12},
        "Średni": {"enabled": True, "topn": 3, "top1_threshold": 0.10},
        "Średnio-Niski": {"enabled": True, "topn": 3, "top1_threshold": 0.08},
        "Bardzo Niski": {"enabled": True, "topn": 4, "top1_threshold": 0.05},
    }
    def __init__(self, assets=GEM_ASSETS, amount=100):
        self.assets = assets
        self.amount = amount

    def run(self):
        scenario, _, _ = TAAStrategy().run()
        config = self.taa_gem_config.get(scenario, {"enabled": True, "topn": 3, "top1_threshold": 0.1})
        if not config["enabled"]:
            print(f"🚨 GEM OFF – scenariusz TAA: {scenario}")
            return {"Cash": self.amount}
        topn = config["topn"]
        thr = config["top1_threshold"]
        scores = {}
        for nm, tkr in self.assets.items():
            data = DataFetcher.fetch_yf(tkr)
            scores[nm] = MomentumCalculator.gem_momentum(data)
        sr = pd.Series(scores).sort_values(ascending=False)
        top = sr[sr > 0].head(topn)
        if top.empty:
            return {"Cash": self.amount}
        if len(top) >= 2 and thr is not None and (top.iloc[0] - top.iloc[1]) > thr:
            return {top.index[0]: self.amount}
        ssum = top.sum()
        return {k: round(v / ssum * self.amount, 2) for k, v in top.items()}

# --- 7) SuperIKEStrategy ---
class SuperIKEStrategy(InvestmentStrategy):
    """
    Strategia SuperIKE oparta na TAA-momentum.
    Domyślnie używane 4 instrumenty z dodatkowym wymuszeniem minimalnej alokacji dla MSCI Poland.
    """
    def __init__(self, amount=100, topn=4, min_msci_poland=15):
        self.amount = amount
        self.topn = topn
        self.min_msci_poland = min_msci_poland
        self.scoring_method = MomentumCalculator.taa_momentum

    def run(self):
        scores = {}
        for name, ticker in SuperIKE_GEM_ETFS.items():
            data = DataFetcher.fetch_yf(ticker)
            score = self.scoring_method(data)
            scores[name] = score
        sr = pd.Series(scores).sort_values(ascending=False)
        top = sr[sr > 0].head(self.topn)
        if top.empty:
            return {"Cash": self.amount}
        total = top.sum()
        allocation = {k: round(v / total * self.amount, 2) for k, v in top.items()}
        # Wymuś minimalną alokację dla MSCI Poland, jeśli jest wybrany
        if "MSCI Poland" in allocation and allocation["MSCI Poland"] < self.min_msci_poland:
            desired = self.min_msci_poland
            allocation["MSCI Poland"] = desired
            other_total = sum(allocation[k] for k in allocation if k != "MSCI Poland")
            for k in allocation:
                if k != "MSCI Poland":
                    allocation[k] = round(allocation[k] * (self.amount - desired) / other_total, 2)
        return allocation

# --- 8) IKZEStrategy ---
class IKZEStrategy(InvestmentStrategy):
    """Strategia IKZE oparta na wykorzystaniu SmartGEMStrategyTAA na IKZE_GEM_ETFS."""
    def run(self):
        s = SmartGEMStrategyTAA(assets=IKZE_GEM_ETFS, amount=100)
        return s.run()

# --- 9) CryptoStrategy ---
class CryptoStrategy(InvestmentStrategy):
    """Jeśli momentum krypto > 0 – 100% Crypto, w przeciwnym razie 100% Cash."""
    def run(self):
        data = DataFetcher.fetch_yf("VBTC.DE")
        sc = MomentumCalculator.taa_momentum(data)
        print(f"Crypto momentum: {sc:.2f}")
        return {"Crypto": 100} if sc > 0 else {"Cash": 100}

# --- 10) DCAStrategy ---
def get_current_pe(ticker: str = "SPY") -> float:
    """
    Pobiera aktualny wskaźnik P/E z yfinance dla podanego ticker'a.
    W przypadku braku danych lub błędu, zwraca 0.0.
    """
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE")
        if pe is None:
            logging.warning(f"P/E dla {ticker} nie jest dostępny; stosuję wartość 0.0")
            return 0.0
        return float(pe)
    except Exception as e:
        logging.error(f"Błąd pobierania P/E dla {ticker}: {e}")
        return 0.0

class DCAStrategy(InvestmentStrategy):
    """Strategia DCA – ustalanie proporcji na podstawie momentum SPY i aktualnego P/E."""
    def run(self):
        spy = DataFetcher.fetch_yf("SPY")
        if spy.empty:
            return "pauza"
        sc = MomentumCalculator.taa_momentum(spy)
        pe = get_current_pe("SPY")
        logging.info(f"Aktualny wskaźnik P/E dla SPY: {pe:.2f}")
        if pe > 30:
            if sc > 0.18:
                return "80/20"
            elif sc > 0.09:
                return "60/40"
            elif sc > 0.045:
                return "40/60"
            elif sc > 0.0225:
                return "20/80"
            else:
                return "pauza"
        else:
            if sc > 0.18:
                return "100/0"
            elif sc > 0.09:
                return "80/20"
            elif sc > 0.045:
                return "60/40"
            elif sc > 0.0225:
                return "40/60"
            elif sc > 0.01125:
                return "20/80"
            else:
                return "pauza"

# --- 11) EDOStrategy ---
class EDOStrategy(InvestmentStrategy):
    """Strategia EDOS – 100% Obligacje EDO."""
    def run(self):
        return {"Obligacje EDO": 100}

# --- 12) SmartGoldenTAA ---
class SmartGoldenTAA(InvestmentStrategy):
    """
    Strategia SmartGoldenTAA (dynamiczny Golden Butterfly):
      - ~60% akcji (VTI)
      - 30% złoto (GLD)
      - 10% obligacje (TLT)
      - 0% Cash (BIL)
      + overlay momentum
    """
    def __init__(self, amount=100):
        self.amount = amount
        self.assets = {"Akcje": "VTI", "Złoto": "GLD", "Obligacje": "TLT", "Cash": "BIL"}
        self.target_alloc = {"Akcje": 0.6, "Złoto": 0.3, "Obligacje": 0.1, "Cash": 0}

    def run(self):
        scores = {}
        for lab, tkr in self.assets.items():
            data = DataFetcher.fetch_yf(tkr)
            sc = MomentumCalculator.taa_momentum(data)
            scores[lab] = sc if sc > 0 else 0
        total = sum(scores.values())
        if total == 0:
            return {"Cash": self.amount}
        final_alloc = {}
        for k in self.target_alloc:
            base = self.target_alloc[k]
            mom = scores.get(k, 0) / total if total > 0 else 0
            w = 0.5 * base + 0.5 * mom
            final_alloc[k] = round(w * self.amount, 2)
        s = sum(final_alloc.values())
        if s > 0:
            final_alloc = {k: round(v / s * 100, 2) for k, v in final_alloc.items()}
        return final_alloc

# ====================================================
# NOWE MODUŁY / WRAPPERY
# ====================================================

# Blok 4.1 – HysteresisDecorator
class HysteresisDecorator(InvestmentStrategy):
    def __init__(self, base_strategy: InvestmentStrategy, threshold=5.0):
        self.base = base_strategy
        self.threshold = threshold  # minimalna zmiana (%) dla aktualizacji sygnału
        self._last_alloc = None
        self._last_extra = ("", "")

    def run(self):
        current = self.base.run()
        if isinstance(current, tuple) and len(current) == 3:
            desc, alloc, alt = current
        else:
            alloc, desc, alt = current, "", ""
        if self._last_alloc is not None:
            diff = sum(
                abs(self._last_alloc.get(a, 0) - alloc.get(a, 0))
                for a in set(self._last_alloc) | set(alloc)
            )
            if diff < self.threshold:
                logging.info(f"Histereza: zmiana {diff:.2f}% < threshold {self.threshold}% – sygnał bez zmian.")
                return (f"{desc} (bez zmiany)", self._last_alloc, self._last_extra)
        self._last_alloc = alloc
        self._last_extra = (desc, alt)
        return (desc, alloc, alt)

# Blok 4.2 – PartialAllocator
class PartialAllocator(InvestmentStrategy):
    def __init__(self, base_strategy: InvestmentStrategy, fraction=0.7, safe_asset="Cash"):
        self.base = base_strategy
        self.fraction = fraction  # np. 0.7 oznacza 70% wg sygnału, pozostałe 30% bezpiecznego aktywa
        self.safe_asset = safe_asset

    def run(self):
        base_result = self.base.run()
        if isinstance(base_result, tuple) and len(base_result) == 3:
            desc, alloc, alt = base_result
        else:
            alloc, desc, alt = base_result, "", ""
        adjusted = {}
        for asset, weight in alloc.items():
            adjusted[asset] = weight * self.fraction
        adjusted[self.safe_asset] = adjusted.get(self.safe_asset, 0) + (1 - self.fraction) * 100
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: round(v / total * 100, 2) for k, v in adjusted.items()}
        return (f"{desc} (partial allocation)", adjusted, alt)

# Blok 4.3 – BlendStrategy
class BlendStrategy(InvestmentStrategy):
    def __init__(self, *strategies: InvestmentStrategy, dynamic_weight_func=None):
        """
        strategies: dowolna liczba strategii do blendowania.
        dynamic_weight_func: opcjonalna funkcja, która przyjmie market_data (dict)
                              i liczbę strategii, a zwróci listę wag o długości
                              równiej liczbie strategii (suma wag = 1).
                              Jeśli nie podana, stosujemy równe wagi.
        """
        self.strategies = strategies
        self.dynamic_weight_func = dynamic_weight_func

    def default_dynamic_weights(self, market_data: dict) -> list:
        # Domyślnie zwracamy równe wagi.
        n = len(self.strategies)
        return [1/n] * n

    def run(self):
        # Przykładowo możemy użyć danych SPY by wpłynąć na wagi.
        market_data = {"SPY": DataFetcher.fetch_yf("SPY")}
        
        # Jeżeli funkcja dynamiczna została przekazana, używamy jej,
        # w przeciwnym razie stosujemy równe wagi.
        weights = (
            self.dynamic_weight_func(market_data, len(self.strategies))
            if self.dynamic_weight_func is not None
            else self.default_dynamic_weights(market_data)
        )
        
        # Upewnij się, że suma wag wynosi 1.
        total_weight = sum(weights)
        if total_weight != 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = self.default_dynamic_weights(market_data)

        # Pobierz wyniki z każdej strategii.
        results = []
        for strat in self.strategies:
            res = strat.run()
            # Zakładamy, że wynik to krotka (desc, alloc, alt) lub alloc (dict)
            if isinstance(res, tuple) and len(res) == 3:
                desc, alloc, alt = res
            else:
                alloc = res
                desc = ""
                alt = ""
            results.append((desc, alloc, alt))
        
        # Blendujemy alokacje – dla każdego aktywa sumujemy wagowane wartości.
        blended = {}
        for (_, alloc, _) in results:
            for asset, pct in alloc.items():
                blended[asset] = blended.get(asset, 0) + pct
        # Przyznajemy każdej strategii jej udział wg dynamicznych wag:
        # Przykładowo, można też blendować alokacje poszczególnych strategii:
        # Inna wersja – sumujemy wagowane alokacje:
        blended = {}
        for (desc, alloc, alt), w in zip(results, weights):
            for asset, pct in alloc.items():
                blended[asset] = blended.get(asset, 0) + pct * w
        total = sum(blended.values())
        if total > 0:
            blended = {k: round(v / total * 100, 1) for k, v in blended.items()}
        
        # Łączymy opisy z poszczególnych strategii.
        descs = " | ".join([res[0] for res in results if res[0]])
        return (f"Blend Strategy: {descs}", blended, tuple(desc for desc, _, _ in results))

# Blok 4.4 – MarketRegimeDetector
class MarketRegimeDetector:
    def __init__(self, vol_window=20, trend_window=200):
        self.vol_window = vol_window
        self.trend_window = trend_window

    def detect_regime(self, market_data: dict) -> str:
        spy_data = market_data.get("SPY")
        if spy_data is None or spy_data.empty:
            return "Neutral"
        spy_price = spy_data.iloc[-1]
        spy_ma = spy_data.rolling(window=self.trend_window).mean().iloc[-1]
        vol = spy_data.pct_change().rolling(window=self.vol_window).std().iloc[-1] * np.sqrt(252)
        if spy_price > spy_ma and vol < 0.15:
            return "Bull-LowVol"
        elif spy_price < spy_ma and vol > 0.25:
            return "Bear-HighVol"
        else:
            return "Neutral"

# Blok 4.5 – SeasonalityFilter
class SeasonalityFilter:
    def adjust_signal(self, allocation: dict) -> dict:
        month = datetime.today().month
        if month in [5, 6, 7, 8, 9, 10]:
            if "Akcje" in allocation:
                original = allocation["Akcje"]
                allocation["Akcje"] = round(original * 0.9, 2)
                allocation["Cash"] = allocation.get("Cash", 0) + round(original * 0.1, 2)
        return allocation

# Blok 4.6 – Rebalance Function
def rebalance_portfolio(target_alloc: dict, current_alloc: dict, tolerance=5.0) -> dict:
    max_diff = 0.0
    for asset in set(target_alloc) | set(current_alloc):
        targ = target_alloc.get(asset, 0)
        curr = current_alloc.get(asset, 0)
        diff = abs(targ - curr)
        if diff > max_diff:
            max_diff = diff
    if max_diff < tolerance:
        logging.info(f"Rebalancing skipped: max drift {max_diff:.2f}% within tolerance {tolerance}%")
        return current_alloc
    else:
        logging.info(f"Rebalancing executed: max drift {max_diff:.2f}% exceeds tolerance {tolerance}%")
        total = sum(target_alloc.values())
        if total > 0:
            return {k: round(v / total * 100, 2) for k, v in target_alloc.items()}
        return target_alloc

# ====================================================
# FINAL WRAPPER – FinalStrategyWrapper
# ====================================================
class FinalStrategyWrapper(InvestmentStrategy):
    def __init__(self, base_strategy: InvestmentStrategy, seasonality=True):
        self.base = base_strategy
        self.seasonality = seasonality
        self.season_filter = SeasonalityFilter()

    def run(self):
        alloc = self.base.run()
        if isinstance(alloc, tuple) and len(alloc) == 3:
            desc, alloc, alt = alloc
        else:
            alloc, desc, alt = alloc, "", ""
        if self.seasonality:
            alloc = self.season_filter.adjust_signal(alloc)
        return (desc + " + sezonowość", alloc, alt)
    
    def name(self):
        return f"FinalStrategyWrapper({self.base.__class__.__name__})"

# Przykładowa strategia finalna
final_wrapped_strategy = FinalStrategyWrapper(
    BlendStrategy(
        TAAStrategy(), 
        PartialAllocator(
            HysteresisDecorator(SmartPriorityTAAStrategy(), threshold=5.0), 
            fraction=0.99, 
            safe_asset="Cash"
        ), 
        PartialAllocator(
            HysteresisDecorator(SmartGEMStrategyTAA(), threshold=5.0), 
            fraction=0.99, 
            safe_asset="Cash"
        ), 
        PartialAllocator(
            HysteresisDecorator(SmartGoldenTAA(), threshold=5.0), 
            fraction=0.99, 
            safe_asset="Cash"
        ), 
        PartialAllocator(
            HysteresisDecorator(SmartPriorityTAAStrategy(), threshold=5.0), 
            fraction=0.99, 
            safe_asset="Cash"
        ), 
        PartialAllocator(
            HysteresisDecorator(TAAStrategy(), threshold=5.0), 
            fraction=0.99, 
            safe_asset="Cash"
        ), 
        PartialAllocator(
            HysteresisDecorator(MomentumTAAStrategy(), threshold=5.0), 
            fraction=0.99, 
            safe_asset="Cash"
        )
    )
)

# ====================================================
# NOWE MIXINY
# ====================================================

class EffectivenessMixin:
    """
    Mierzy skuteczność strategii na przestrzeni 12 miesięcy (CAGR, Sharpe, Max Drawdown)
    i porównuje wynik do benchmarku (np. SPY).
    """
    def get_ticker_for_asset(self, asset: str) -> str:
        mapping = {
            "SPY": "SPY",
            "S&P 500": "SPY",
            "USA (SPY)": "SPY",
            "Akcje": "SPY",
            "Obligacje": "TLT",
            "Złoto": "GLD",
            "Gold": "GLD",
            "Gold (GLD)": "GLD",
            "Cash": "BIL",
            "Cash": "BIL",
            "Cash (BIL)": "BIL",
            "ACWI ex-US": "ACWX",
            "ex-US (ACWX)": "ACWX",
            "Emerging": "EEM",
            "Crypto": "VBTC.DE",
            "Obligacje EDO": "Obligacje EDO",
            "Beta CASH": "ETFBCASH.WA",
            "TBSP": "ETFBTBSP.WA",
            "MSCI Poland": "EPOL",
            "Amundi DAX (ACWX)": "ACWX",
        }
        return mapping.get(asset, asset)
    
    def check_effectiveness(self, benchmark="SPY", period_months=12, dd_threshold=0.20):
        dates = pd.date_range(end=datetime.today(), periods=period_months+1, freq='M')
        strat_returns = []
        for i in range(len(dates)-1):
            alloc = self.run()  # zakładamy, że wynik to dict
            month_return = 0.0
            for asset, pct in alloc.items():
                ticker = self.get_ticker_for_asset(asset)
                if ticker is None:
                    r = 0.0
                else:
                    data = DataFetcher.fetch_yf(ticker)
                    if data.empty:
                        r = 0.0
                    else:
                        try:
                            price_start = float(data.iloc[0])
                            price_end = float(data.iloc[-1])
                            r = (price_end / price_start) - 1
                        except Exception:
                            r = 0.0
                month_return += (pct / 100.0) * r
            strat_returns.append(month_return)
        cum_returns = np.cumprod([1 + r for r in strat_returns])
        cagr = cum_returns[-1] ** (12 / period_months) - 1
        sharpe = (np.mean(strat_returns) * np.sqrt(12)) / (np.std(strat_returns) or 1)
        peak = np.maximum.accumulate(cum_returns)
        dd = abs(np.min((cum_returns - peak) / peak))
        status = "✅" if (cagr > 0.05 and sharpe > 0.5 and dd < dd_threshold) else "❌"
        msg = (f"{self.__class__.__name__}: {status} CAGR {cagr*100:.1f}% vs. SPY, "
               f"Sharpe {sharpe:.2f}, Max DD {dd*100:.1f}%")
        return msg

class SidewaysDetector:
    """
    Wykrywa boczniaka na podstawie zakresu cen i zmienności (np. dla SPY).
    """
    def is_sideways_market(self, ticker="SPY", range_window_months=6, threshold_range=8.0, threshold_vol=2.5):
        data = DataFetcher.fetch_yf(ticker)
        if data.empty:
            return (False, f"{ticker}: Brak danych.")
        price_max = float(data.max())
        price_min = float(data.min())
        rng = (price_max - price_min) / price_min * 100
        vol = float(data.pct_change().std() * (252 ** 0.5) * 100)
        detected = (rng < threshold_range and vol < threshold_vol)
        msg = f"{ticker}: Sideways detected – {range_window_months}M range = {rng:.1f}%, volatility = {vol:.1f}%"
        return (detected, msg)

class SoftStopMixin:
    """
    Jeśli momentum (dla SPY) jest ujemne, wymusza 100% alokację w Cash.
    """
    def soft_stop(self):
        data = DataFetcher.fetch_yf("SPY")
        if data.empty:
            return False
        mom = (float(data.iloc[-1]) / float(data.iloc[-2])) - 1
        return mom < 0

# --- Enhanced Strategy łącząca bazową strategię z mixinami ---
class EnhancedSmartStrategy(EffectivenessMixin, SidewaysDetector, SoftStopMixin, InvestmentStrategy):
    """
    Rozszerzona strategia: uruchamia bazową strategię,
    ale – jeśli momentum spadkowe (soft stop) – zwraca 100% Cash,
    dodatkowo dostarcza benchmarking i wykrywanie boczniaka.
    """
    def __init__(self, base_strategy: InvestmentStrategy):
        self.base_strategy = base_strategy

    def run(self):
        if self.soft_stop():
            return {"Cash": 100}
        return self.base_strategy.run()

    def name(self):
        return f"Enhanced {self.base_strategy.__class__.__name__}"

# ====================================================
# APLIKACYJNA WARSTWA – MultiStrategyPortfolio, CQRS, Event Bus, ReportBuilder, StrategyRunner
# ====================================================
def limit_allocation(alloc: dict, min_pct=5, max_pct=85) -> dict:
    limited = {}
    for key, pct in alloc.items():
        if pct < min_pct:
            limited[key] = min_pct
        elif pct > max_pct:
            limited[key] = max_pct
        else:
            limited[key] = pct
    total = sum(limited.values())
    if total > 0:
        return {k: round(v / total * 100, 2) for k, v in limited.items()}
    return alloc

class MultiStrategyPortfolio(InvestmentStrategy):
    """
    Łączy wiele strategii wg określonych wag.
    Po zsumowaniu wyników stosuje limit_allocation (min. 5%, max. 85%).
    """
    def __init__(self, strategies_with_weights):
        self.strategies_with_weights = strategies_with_weights
        total_w = sum(w for _, w in self.strategies_with_weights)
        if abs(total_w - 1.0) > 0.0001:
            raise ValueError(f"Sum of weights != 1.0 (jest {total_w})")

    def run(self):
        combined = {}
        for strat, weight in self.strategies_with_weights:
            alloc = strat.run()
            if not isinstance(alloc, dict):
                if isinstance(alloc, tuple) and len(alloc) == 3:
                    _, alloc, _ = alloc
                else:
                    continue
            for asset, pct in alloc.items():
                combined[asset] = combined.get(asset, 0) + pct * weight
        total = sum(combined.values())
        if total > 0:
            combined = {k: round(v / total * 100, 2) for k, v in combined.items()}
        combined = limit_allocation(combined, 5, 85)
        return combined

# CQRS – definicje: zdarzenie, komenda, query, Event Bus, ReportBuilder, StrategyRunner

class StrategyExecutedEvent:
    """Zdarzenie wykonania strategii."""
    def __init__(self, strat_name: str, result: dict):
        self.strat_name = strat_name
        self.result = result
        self.timestamp = datetime.now()

class ExecuteStrategyCommand:
    """Komenda wykonania strategii."""
    def __init__(self, strategy: InvestmentStrategy):
        self.strategy = strategy

class StrategyCommandHandler:
    """Handler wykonania komendy, który publikuje zdarzenie."""
    def __init__(self, event_bus):
        self.event_bus = event_bus

    def handle(self, command: ExecuteStrategyCommand) -> dict:
        res = command.strategy.run()
        event = StrategyExecutedEvent(command.strategy.name(), res)
        self.event_bus.publish(event)
        return res

class GetStrategyReportQuery:
    """Zapytanie o raport strategii."""
    def __init__(self, strategy: InvestmentStrategy):
        self.strategy = strategy

class StrategyQueryHandler:
    """Handler zapytania, budujący raport przy użyciu ReportBuilder."""
    def handle(self, query: GetStrategyReportQuery) -> str:
        res = query.strategy.run()
        report = ReportBuilder().build_report(query.strategy.name(), res)
        return report

class EventBus:
    """Prosty Event Bus do subskrybowania zdarzeń."""
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, evt_type, handler):
        if evt_type not in self.subscribers:
            self.subscribers[evt_type] = []
        self.subscribers[evt_type].append(handler)

    def publish(self, event):
        for handler in self.subscribers.get(type(event), []):
            handler(event)

def log_strategy_execution(event: StrategyExecutedEvent):
    logging.info(f"Strategia '{event.strat_name}' wykonana {event.timestamp}. Wynik: {event.result}")

class ReportBuilder:
    """Budowniczy raportu strategii."""
    def __init__(self):
        self.sections = []

    def add_section(self, title: str):
        self.sections.append(f"\n=== {title} ===")

    def add_tuple_result(self, desc: str, alloc: dict, alt: tuple):
        self.sections.append(f"Opis: {desc}")
        for k, v in alloc.items():
            self.sections.append(f"- {k}: {v}%")
        if alt != ("-", "-"):
            self.sections.append(f"Alternatywa: {alt[0]}, {alt[1]}")

    def add_dict_result(self, alloc: dict):
        for k, v in alloc.items():
            self.sections.append(f"- {k}: {v}%")

    def add_string(self, text: str):
        self.sections.append(str(text))

    def build_report(self, strat_name: str, res: dict) -> str:
        self.add_section(f"Raport strategii: {strat_name}")
        if isinstance(res, dict):
            self.add_dict_result(res)
        else:
            self.add_string(str(res))
        return "\n".join(self.sections)

class StrategyRunner:
    """Fasada do uruchamiania strategii i budowania raportu."""
    def __init__(self):
        self.builder = ReportBuilder()

    def run_and_log(self, title: str, strategy: InvestmentStrategy):
        self.builder.add_section(title)
        res = strategy.run()
        if isinstance(res, tuple) and len(res) == 3:
            desc, alloc, alt = res
            self.builder.add_tuple_result(desc, alloc, alt)
        elif isinstance(res, dict):
            self.builder.add_dict_result(res)
        elif isinstance(res, str):
            self.builder.add_string(res)
        else:
            self.builder.add_string(str(res))

    def get_report(self) -> str:
        return self.builder.build_report("Final", {})

# ====================================================
# MAIN – Orkiestracja i przykładowe uruchomienie strategii
# ====================================================
if __name__ == "__main__":
    runner = StrategyRunner()

    # Lista strategii do przetestowania
    all_strats = [

        ("DCA", DCAStrategy()),
        ("IKZE Żony", EDOStrategy()),
        ("Crypto", CryptoStrategy()),        
        ("SuperIKE GEM", SuperIKEStrategy()),        
        ("Enchanced SuperIKE", EnhancedSmartStrategy(SuperIKEStrategy())),
        #("Enchanced DCAStrategy", EnhancedSmartStrategy(DCAStrategy())),
        #("Enchanced TAAStrategy", EnhancedSmartStrategy(TAAStrategy())),
        ("Enchanced MomentumTAAStrategy", EnhancedSmartStrategy(MomentumTAAStrategy())),
        ("TAA (Heurystyka)", TAAStrategy()),
        ("TAA (Momentum)", MomentumTAAStrategy()),
        ("TAA (Smart Priority)", SmartPriorityTAAStrategy()),
        ("TAA (SmartGoldenTAA)", SmartGoldenTAA()),
        ("SmartGEMStrategyTAA", SmartGEMStrategyTAA()),
        # ("Enchanced SmartPriorityTAAStrategy", EnhancedSmartStrategy(SmartPriorityTAAStrategy())),
        # ("Enchanced SmartGoldenTAA", EnhancedSmartStrategy(SmartGoldenTAA())),
        #("Enchanced SmartGEMStrategyTAA", EnhancedSmartStrategy(SmartGEMStrategyTAA())),
        #("Enchanced Crypto", EnhancedSmartStrategy(CryptoStrategy())),        
        ("Final Wrapped Strategy", final_wrapped_strategy)
    ]

    for title, strat in all_strats:
        runner.run_and_log(title, strat)

    final_report = runner.builder.build_report("Raport Końcowy", {})
    print(f"\n=== STRATEGIA IT-TATA (DDD + Wzorce) [{TODAY}] ===\n")
    print(final_report)
    print("\n=== KONIEC ===\n")
    
    # Przykładowe użycie CQRS / Event Bus
    # event_bus = EventBus()
    # event_bus.subscribe(StrategyExecutedEvent, log_strategy_execution)
    # command_handler = StrategyCommandHandler(event_bus)
    # cmd = ExecuteStrategyCommand(final_wrapped_strategy)
    # command_handler.handle(cmd)


    my_strategies = [
        (DCAStrategy(), 0.2025),
        (SmartGoldenTAA(), 0.15),
        (SmartGEMStrategyTAA(), 0.145),
        (CryptoStrategy(), 0.0025),
        (EDOStrategy(), 0.075),
        (EDOStrategy(), 0.05),
        (EDOStrategy(), 0.25),
        (SuperIKEStrategy(), 0.125)
    ]
    multi = MultiStrategyPortfolio(my_strategies)
    runner.run_and_log("Core-Satellite (SmartGoldenTAA + SmartGEMStrategyTAA)", multi)
    
    final_report = runner.builder.build_report("Final Report", {})
    print(f"\n=== STRATEGIA IT-TATA (DDD + Wzorce) [{TODAY}] ===\n")
    print(final_report)
    print("\n=== KONIEC ===\n")
    
    # my_strategies = [
    #     (EnhancedSmartStrategy(DCAStrategy()), 0.3),
    #     (EnhancedSmartStrategy(SmartGoldenTAA()), 0.25),
    #     (EnhancedSmartStrategy(SmartGEMStrategyTAA()), 0.245),
    #     (EnhancedSmartStrategy(CryptoStrategy()), 0.005),
    #     (EnhancedSmartStrategy(SuperIKEStrategy()), 0.2)
    # ]
    
    # multi = MultiStrategyPortfolio(my_strategies)
    # runner.run_and_log("Core-Satellite (Enhanced: SmartGoldenTAA + SmartGEMStrategyTAA)", multi)
    
    # final_report = runner.builder.build_report("Final Report", {})
    # print(f"\n=== STRATEGIA IT-TATA (DDD + Wzorce) [{TODAY}] ===\n")
    # print(final_report)
    # print("\n=== KONIEC ===\n")
    
    # Test Enhanced Strategy z mixinami
    base_strat = SuperIKEStrategy()
    enhanced = EnhancedSmartStrategy(base_strat)
    if enhanced.soft_stop():
        print("SoftStopMixin: Momentum ujemne → 100% Cash")
    else:
        print("SoftStopMixin: Momentum dodatnie → Kontynuuj zakupy")
    eff_msg = enhanced.check_effectiveness(benchmark="SPY")
    print(eff_msg)
    sideways, sw_msg = enhanced.is_sideways_market(ticker="SPY", range_window_months=6)
    print(sw_msg)
    logging.info("Tracing/Monitoring: Test Enhanced Strategy zakończony.")


    # Parametry:
    # Cotygodniowa_kwota: 1250 PLN
    # Strategie:
    # TAA:   45%   # 562,50 PLN 150 usd 600 usd
    # GEM:   27,5% # 343,75 PLN 80 eur 320 eur
    # DCA:   25%   # 312,50 PLN 75 eur 300 eur
    # AEM:   2,5%  # 31,25 PLN  15 eur 60 eur
    # Dni_offset: [0, 7, 14, 21, 28]  # od 20.
    # Godzina: 11:00