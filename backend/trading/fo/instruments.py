"""
Instrument master loader for F&O mapping.
Loads broker instrument CSV and resolves futures contracts.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, date
from typing import Dict, List, Optional, Set


class InstrumentMaster:
    def __init__(self, path: Optional[str] = None):
        self.path = path or ""
        self._rows: List[Dict] = []
        self._loaded = False
        if self.path and os.path.exists(self.path):
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._rows.append(self._normalize(row))
            self._loaded = True
        except Exception:
            self._rows = []
            self._loaded = False

    @staticmethod
    def _normalize(row: Dict) -> Dict:
        return {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

    @staticmethod
    def _get(row: Dict, keys: List[str]) -> Optional[str]:
        for key in keys:
            if key in row and row[key] not in [None, ""]:
                return str(row[key])
        return None

    @staticmethod
    def _parse_expiry(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except Exception:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except Exception:
            return None

    @staticmethod
    def _infer_underlying(tradingsymbol: str) -> str:
        if not tradingsymbol:
            return ""
        m = re.match(r"^[A-Z]+", tradingsymbol)
        return m.group(0) if m else tradingsymbol

    @staticmethod
    def _is_futures(row: Dict) -> bool:
        itype = (row.get("instrument_type") or row.get("instrumenttype") or "").upper()
        segment = (row.get("segment") or row.get("exchange_segment") or row.get("exchangecode") or "").upper()
        ts = (row.get("tradingsymbol") or row.get("symbol") or row.get("trading_symbol") or "").upper()
        if "FUT" in itype:
            return True
        if "FUT" in segment:
            return True
        if ts.endswith("FUT"):
            return True
        return False

    def available(self) -> bool:
        return self._loaded and len(self._rows) > 0

    def get_fo_symbols(self) -> Set[str]:
        if not self.available():
            return set()
        symbols: Set[str] = set()
        for row in self._rows:
            if not self._is_futures(row):
                continue
            underlying = self._get(row, ["underlying", "name", "underlyingsymbol"])
            tradingsymbol = self._get(row, ["tradingsymbol", "trading_symbol", "symbol"])
            if not tradingsymbol and not underlying:
                continue
            if not underlying:
                underlying = self._infer_underlying(tradingsymbol.upper())
            if underlying:
                symbols.add(underlying)
        return symbols

    def get_futures_contract(self, underlying: str, on_date: Optional[date] = None) -> Optional[Dict]:
        """
        Return nearest futures contract row for an underlying symbol.
        """
        if not self.available():
            return None
        target = underlying.upper()
        on_date = on_date or date.today()

        candidates: List[Dict] = []
        for row in self._rows:
            if not self._is_futures(row):
                continue
            underlying = self._get(row, ["underlying", "name", "underlyingsymbol"])
            tradingsymbol = self._get(row, ["tradingsymbol", "trading_symbol", "symbol"])
            if not tradingsymbol and not underlying:
                continue
            if underlying and underlying.upper() != target:
                continue
            if tradingsymbol and not tradingsymbol.upper().startswith(target):
                continue

            expiry_raw = self._get(row, ["expiry", "expiry_date", "exp_date"])
            expiry = self._parse_expiry(expiry_raw)
            if expiry and expiry >= on_date:
                candidates.append({
                    "tradingsymbol": tradingsymbol,
                    "exchange": (self._get(row, ["exchange", "exch", "segment"]) or "NFO").upper(),
                    "expiry": expiry,
                    "lot_size": self._safe_int(self._get(row, ["lot_size", "lotsize", "lot"])),
                    "instrument_token": self._get(row, ["instrument_token", "instrument_key", "token", "instrumenttoken"]),
                })

        if not candidates:
            return None
        candidates.sort(key=lambda x: x.get("expiry"))
        return candidates[0]

    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except Exception:
            return None
