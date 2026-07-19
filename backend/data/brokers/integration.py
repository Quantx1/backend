"""
================================================================================
                    QUANT X BROKER INTEGRATION
                    ===========================

    Supports: Zerodha, Angel One, Upstox, Fyers (beta), Dhan (beta),
              Kotak Neo (beta), Alice Blue (beta)
================================================================================
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    CNC = "CNC"
    MIS = "MIS"
    NRML = "NRML"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    symbol: str
    exchange: str
    transaction_type: TransactionType
    quantity: int
    product: ProductType
    order_type: OrderType
    price: float = 0
    trigger_price: float = 0
    instrument_token: Optional[str] = None
    order_id: str = None
    status: OrderStatus = None
    filled_quantity: int = 0
    average_price: float = 0
    message: str = ""


@dataclass
class Position:
    symbol: str
    exchange: str
    quantity: int
    average_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    product: ProductType


@dataclass
class GTTOrder:
    symbol: str
    exchange: str
    trigger_type: str
    trigger_values: List[float]
    orders: List[Dict]
    gtt_id: str = None
    status: str = None


class BaseBroker(ABC):
    def __init__(self, credentials: Dict):
        self.credentials = credentials
        self.access_token = None
        self.is_authenticated = False
        self.name = "BaseBroker"

    @abstractmethod
    def login(self) -> bool: pass

    @abstractmethod
    def place_order(self, order: Order) -> Order: pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus: pass

    def get_orders(self) -> List[Dict]:
        """
        Return the user's order book as a normalized list of dicts with keys:
        order_id, symbol, transaction_type, quantity, filled_quantity,
        order_type, price, average_price, status, product.
        Default returns []; override in subclass for broker-specific fetch.
        """
        return []

    @abstractmethod
    def get_positions(self) -> List[Position]: pass

    @abstractmethod
    def get_holdings(self) -> List[Dict]: pass

    @abstractmethod
    def get_quote(self, symbol: str, exchange: str) -> Dict: pass

    @abstractmethod
    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder: pass

    @abstractmethod
    def get_available_margin(self) -> float: pass

    def get_option_chain(self, symbol: str, expiry: str = "") -> List[Dict]:
        """
        Fetch live options chain from broker API.
        Returns list of dicts with keys: strike, option_type ('CE'/'PE'),
        ltp, bid, ask, oi, oi_change, volume, iv, delta, gamma, theta, vega.
        Override in subclass for broker-specific implementation.
        """
        return []


class ZerodhaBroker(BaseBroker):
    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "Zerodha"
        self.kite = None
        self._enctoken = None  # Direct API mode (no KiteConnect library)
        self._session = None

    def login(self) -> bool:
        try:
            # Mode 1: enctoken (direct OMS API — no KiteConnect library needed)
            if 'enctoken' in self.credentials:
                import requests as _requests
                self._enctoken = self.credentials['enctoken']
                self._session = _requests.Session()
                self._session.headers.update({
                    "Authorization": f"enctoken {self._enctoken}",
                    "X-Kite-Version": "3",
                })
                # Verify token is valid
                resp = self._session.get(
                    "https://kite.zerodha.com/oms/user/profile", timeout=10
                )
                if resp.status_code == 200:
                    self.is_authenticated = True
                    logger.info("Zerodha enctoken auth OK")
                    return True
                logger.warning(f"Zerodha enctoken auth failed: {resp.status_code}")
                return False

            # Mode 2: KiteConnect access_token (standard OAuth)
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.credentials['api_key'])
            if 'access_token' in self.credentials:
                self.kite.set_access_token(self.credentials['access_token'])
                self.is_authenticated = True
                return True
            return False
        except Exception as e:
            logger.error(f"Zerodha login error: {e}")
            return False

    def _oms_post(self, path: str, data: Dict) -> Dict:
        """Direct OMS API call using enctoken."""
        resp = self._session.post(
            f"https://kite.zerodha.com/oms{path}", data=data, timeout=15
        )
        return resp.json()

    def _oms_get(self, path: str, params: Dict = None) -> Dict:
        """Direct OMS API GET using enctoken."""
        resp = self._session.get(
            f"https://kite.zerodha.com/oms{path}", params=params, timeout=15
        )
        return resp.json()

    def _oms_delete(self, path: str) -> Dict:
        """Direct OMS API DELETE using enctoken."""
        resp = self._session.delete(
            f"https://kite.zerodha.com/oms{path}", timeout=15
        )
        return resp.json()

    def place_order(self, order: Order) -> Order:
        if not self.is_authenticated:
            order.status = OrderStatus.REJECTED
            order.message = "Not authenticated"
            return order

        order_data = {
            "tradingsymbol": order.symbol,
            "exchange": order.exchange,
            "transaction_type": order.transaction_type.value,
            "quantity": order.quantity,
            "product": order.product.value,
            "order_type": order.order_type.value,
        }
        if order.order_type == OrderType.LIMIT:
            order_data["price"] = order.price
        if order.order_type in [OrderType.SL, OrderType.SL_M]:
            order_data["trigger_price"] = order.trigger_price
        order_data["validity"] = "DAY"

        try:
            # Enctoken direct API
            if self._enctoken:
                result = self._oms_post("/orders/regular", order_data)
                if result.get("status") == "success":
                    order.order_id = str(result["data"]["order_id"])
                    order.status = OrderStatus.PENDING
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = result.get("message", "Unknown error")
                return order

            # KiteConnect library
            order_id = self.kite.place_order(
                variety="regular",
                tradingsymbol=order.symbol,
                exchange=order.exchange,
                transaction_type=order.transaction_type.value,
                quantity=order.quantity,
                product=order.product.value,
                order_type=order.order_type.value,
                price=order.price if order.order_type == OrderType.LIMIT else None,
                trigger_price=order.trigger_price if order.order_type in [OrderType.SL, OrderType.SL_M] else None
            )
            order.order_id = str(order_id)
            order.status = OrderStatus.PENDING
            return order
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            if self._enctoken:
                result = self._oms_delete(f"/orders/regular/{order_id}")
                return result.get("status") == "success"
            self.kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except BaseException:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            if self._enctoken:
                result = self._oms_get("/orders")
                for o in result.get("data", []):
                    if str(o['order_id']) == order_id:
                        return OrderStatus[o['status'].upper()]
                return OrderStatus.PENDING
            for o in self.kite.orders():
                if str(o['order_id']) == order_id:
                    return OrderStatus[o['status'].upper()]
            return OrderStatus.PENDING
        except BaseException:
            return OrderStatus.PENDING

    def get_orders(self) -> List[Dict]:
        """Normalized order book (reuses the same fetch as get_order_status)."""
        try:
            if self._enctoken:
                raw = self._oms_get("/orders").get("data", []) or []
            else:
                raw = self.kite.orders() or []
            orders = []
            for o in raw:
                orders.append({
                    "order_id": str(o.get("order_id", "")),
                    "symbol": o.get("tradingsymbol", ""),
                    "transaction_type": str(o.get("transaction_type", "")).upper(),
                    "quantity": int(o.get("quantity", 0) or 0),
                    "filled_quantity": int(o.get("filled_quantity", 0) or 0),
                    "order_type": str(o.get("order_type", "")).upper(),
                    "price": float(o.get("price", 0) or 0),
                    "average_price": float(o.get("average_price", 0) or 0),
                    "status": str(o.get("status", "")).upper(),
                    "product": o.get("product", "") or "",
                })
            # Kite returns oldest-first; reverse to keep newest-first.
            orders.reverse()
            return orders
        except BaseException:
            return []

    def get_positions(self) -> List[Position]:
        positions = []
        try:
            if self._enctoken:
                result = self._oms_get("/portfolio/positions")
                net = result.get("data", {}).get("net", [])
            else:
                net = self.kite.positions().get('net', [])
            for p in net:
                if p['quantity'] != 0:
                    positions.append(Position(
                        symbol=p['tradingsymbol'],
                        exchange=p['exchange'],
                        quantity=p['quantity'],
                        average_price=p['average_price'],
                        current_price=p['last_price'],
                        pnl=p['pnl'],
                        pnl_percent=(p['pnl'] / (p['average_price'] * abs(p['quantity']))) * 100 if p['average_price'] > 0 else 0,
                        product=ProductType.CNC
                    ))
        except Exception as e:
            logger.error(f"Positions error: {e}")
        return positions

    def get_holdings(self) -> List[Dict]:
        try:
            if self._enctoken:
                result = self._oms_get("/portfolio/holdings")
                return result.get("data", [])
            return self.kite.holdings()
        except BaseException:
            return []

    def get_quote(self, symbol: str, exchange: str) -> Dict:
        try:
            if self._enctoken:
                result = self._oms_get("/quote", {"i": f"{exchange}:{symbol}"})
                return result.get("data", {}).get(f"{exchange}:{symbol}", {})
            return self.kite.quote([f"{exchange}:{symbol}"]).get(f"{exchange}:{symbol}", {})
        except BaseException:
            return {}

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        try:
            if self._enctoken:
                # GTT/OCO isn't available on the enctoken OMS, but a single-leg
                # SL-M order IS — place it so the position still gets a REAL
                # broker-side stop (survives a Quant X outage). The target leg is
                # managed by the position-monitor cron, same as the Upstox/Angel
                # SL-M path. Cancellable via the regular cancel_order on close.
                # Equity (CNC) only; F&O (NRML) enctoken stops aren't covered here
                # and stay alert-on-unprotected (no regression).
                sl_leg = gtt.orders[0] if gtt.orders else {}
                sl_order = Order(
                    symbol=gtt.symbol,
                    exchange=gtt.exchange,
                    transaction_type=TransactionType(sl_leg.get("transaction_type", "SELL")),
                    quantity=int(sl_leg.get("quantity") or 0),
                    product=ProductType.CNC,
                    order_type=OrderType.SL_M,
                    trigger_price=float(gtt.trigger_values[0]),
                )
                placed = self.place_order(sl_order)
                if placed.status == OrderStatus.REJECTED:
                    gtt.status = "sl_failed"
                    logger.warning("Zerodha enctoken SL-M stop rejected: %s", placed.message)
                else:
                    gtt.gtt_id = placed.order_id
                    gtt.status = "sl_placed"
                    logger.info("Zerodha enctoken SL-M stop placed: %s", placed.order_id)
                return gtt
            gtt_id = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_OCO if gtt.trigger_type == "two-leg" else self.kite.GTT_TYPE_SINGLE,
                tradingsymbol=gtt.symbol,
                exchange=gtt.exchange,
                trigger_values=gtt.trigger_values,
                last_price=gtt.orders[0].get('last_price', gtt.trigger_values[0]),
                orders=gtt.orders
            )
            gtt.gtt_id = str(gtt_id)
            gtt.status = "active"
        except Exception as e:
            gtt.status = "failed"
            logger.error(f"GTT error: {e}")
        return gtt

    def get_available_margin(self) -> float:
        try:
            if self._enctoken:
                result = self._oms_get("/user/margins")
                return result.get("data", {}).get("equity", {}).get("available", {}).get("live_balance", 0)
            return self.kite.margins().get('equity', {}).get('available', {}).get('live_balance', 0)
        except BaseException:
            return 0

    def get_option_chain(self, symbol: str, expiry: str = "") -> List[Dict]:
        """Fetch options chain via Kite Connect instruments + quote API."""
        if not self.is_authenticated or not self.kite:
            return []
        try:
            # Build instrument keys for NFO segment
            instruments = self.kite.instruments("NFO")
            filtered = [
                i for i in instruments
                if i['name'] == symbol
                and i['instrument_type'] in ('CE', 'PE')
                and (not expiry or str(i['expiry']) == expiry)
            ]
            if not filtered:
                return []

            # Pick nearest expiry if not specified
            if not expiry:
                expiries = sorted(set(i['expiry'] for i in filtered))
                nearest = expiries[0] if expiries else None
                if not nearest:
                    return []
                filtered = [i for i in filtered if i['expiry'] == nearest]

            # Fetch quotes in batches of 200 (Kite limit)
            chain = []
            batch_size = 200
            for i in range(0, len(filtered), batch_size):
                batch = filtered[i:i + batch_size]
                keys = [f"NFO:{inst['tradingsymbol']}" for inst in batch]
                quotes = self.kite.quote(keys)

                for inst in batch:
                    key = f"NFO:{inst['tradingsymbol']}"
                    q = quotes.get(key, {})
                    q.get('ohlc', {})
                    chain.append({
                        'strike': float(inst['strike']),
                        'option_type': inst['instrument_type'],  # 'CE' or 'PE'
                        'expiry': str(inst['expiry']),
                        'ltp': q.get('last_price', 0),
                        'bid': q.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                        'ask': q.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                        'oi': q.get('oi', 0),
                        'oi_change': q.get('oi_day_high', 0) - q.get('oi_day_low', 0),
                        'volume': q.get('volume', 0),
                        'iv': 0,  # Kite doesn't return IV directly; computed downstream
                        'lot_size': inst.get('lot_size', 1),
                        'tradingsymbol': inst['tradingsymbol'],
                    })
            return chain
        except Exception as e:
            logger.error(f"Zerodha option chain error for {symbol}: {e}")
            return []


class AngelOneBroker(BaseBroker):
    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "AngelOne"
        self.smart_api = None
        self._refresh_token = None

    def login(self) -> bool:
        try:
            from SmartApi import SmartConnect
            import pyotp
            self.smart_api = SmartConnect(api_key=self.credentials['api_key'])
            totp = pyotp.TOTP(self.credentials['totp_secret']).now()
            data = self.smart_api.generateSession(
                clientCode=self.credentials['client_id'],
                password=self.credentials['password'],
                totp=totp
            )
            if data['status']:
                self.access_token = data['data']['jwtToken']
                self._refresh_token = data['data'].get('refreshToken')
                self.is_authenticated = True
                return True
            return False
        except Exception as e:
            logger.error(f"AngelOne login error: {e}")
            return False

    def refresh_session(self) -> bool:
        """Refresh expired session using refresh token."""
        try:
            if not self._refresh_token:
                return self.login()
            data = self.smart_api.generateToken(self._refresh_token)
            if data['status']:
                self.access_token = data['data']['jwtToken']
                self._refresh_token = data['data'].get('refreshToken', self._refresh_token)
                self.is_authenticated = True
                return True
            return self.login()
        except Exception:
            return self.login()

    def place_order(self, order: Order) -> Order:
        if not self.is_authenticated:
            order.status = OrderStatus.REJECTED
            return order
        try:
            response = self.smart_api.placeOrder({
                "variety": "NORMAL",
                "tradingsymbol": order.symbol,
                "transactiontype": order.transaction_type.value,
                "exchange": order.exchange,
                "ordertype": order.order_type.value,
                "producttype": "DELIVERY" if order.product == ProductType.CNC else "INTRADAY",
                "duration": "DAY",
                "quantity": str(order.quantity),
                "price": str(order.price) if order.order_type == OrderType.LIMIT else "0"
            })
            if response['status']:
                order.order_id = response['data']['orderid']
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
                order.message = response['message']
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            return self.smart_api.cancelOrder(order_id, "NORMAL")['status']
        except BaseException:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            for o in self.smart_api.orderBook()['data']:
                if o['orderid'] == order_id:
                    return OrderStatus[o['orderstatus'].upper()]
            return OrderStatus.PENDING
        except BaseException:
            return OrderStatus.PENDING

    def get_orders(self) -> List[Dict]:
        """Normalized order book (reuses the same fetch as get_order_status)."""
        try:
            book = self.smart_api.orderBook()
            raw = (book.get("data") or []) if isinstance(book, dict) else []
            orders = []
            for o in raw:
                # Angel exposes status as 'orderstatus' (preferred) or 'status'.
                status = o.get("orderstatus") or o.get("status") or ""
                orders.append({
                    "order_id": str(o.get("orderid", "")),
                    "symbol": o.get("tradingsymbol", ""),
                    "transaction_type": str(o.get("transactiontype", "")).upper(),
                    "quantity": int(float(o.get("quantity", 0) or 0)),
                    "filled_quantity": int(float(o.get("filledshares", 0) or 0)),
                    "order_type": str(o.get("ordertype", "")).upper(),
                    "price": float(o.get("price", 0) or 0),
                    "average_price": float(o.get("averageprice", 0) or 0),
                    "status": str(status).upper(),
                    "product": o.get("producttype", "") or "",
                })
            return orders
        except BaseException:
            return []

    def get_positions(self) -> List[Position]:
        positions = []
        try:
            response = self.smart_api.position()
            if response['status']:
                for p in response['data']:
                    if int(p['netqty']) != 0:
                        positions.append(Position(
                            symbol=p['tradingsymbol'],
                            exchange=p['exchange'],
                            quantity=int(p['netqty']),
                            average_price=float(p['averageprice']),
                            current_price=float(p['ltp']),
                            pnl=float(p['pnl']),
                            pnl_percent=0,
                            product=ProductType.CNC
                        ))
        except Exception as e:
            logger.error(f"Positions error: {e}")
        return positions

    def get_holdings(self) -> List[Dict]:
        try:
            r = self.smart_api.holding()
            return r['data'] if r['status'] else []
        except BaseException:
            return []

    def get_quote(self, symbol: str, exchange: str) -> Dict:
        try:
            r = self.smart_api.ltpData(exchange, symbol, "")
            return r['data'] if r['status'] else {}
        except BaseException:
            return {}

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        """
        Angel One doesn't support native GTT via SDK.
        Alternative: Place SL-M order for stop loss protection.
        Target is managed by the position monitor (scheduler).
        """
        if not self.is_authenticated or not gtt.trigger_values:
            gtt.status = "failed"
            return gtt
        try:
            sl_price = gtt.trigger_values[0]
            sell_order = gtt.orders[0] if gtt.orders else {}
            qty = sell_order.get('quantity', 0)
            if qty <= 0:
                gtt.status = "failed"
                return gtt
            response = self.smart_api.placeOrder({
                "variety": "STOPLOSS",
                "tradingsymbol": gtt.symbol,
                "transactiontype": sell_order.get('transaction_type', 'SELL'),
                "exchange": gtt.exchange,
                "ordertype": "STOPLOSS_MARKET",
                "producttype": "DELIVERY",
                "duration": "DAY",
                "quantity": str(qty),
                "triggerprice": str(sl_price),
                "price": "0",
            })
            if response.get('status'):
                gtt.gtt_id = response['data'].get('orderid', '')
                gtt.status = "sl_placed"
                logger.info(f"AngelOne SL-M order placed for {gtt.symbol} at {sl_price}")
            else:
                gtt.status = "sl_failed"
                logger.warning(f"AngelOne SL-M failed: {response.get('message', '')}")
        except Exception as e:
            gtt.status = "sl_failed"
            logger.error(f"AngelOne GTT alternative error: {e}")
        return gtt

    def get_available_margin(self) -> float:
        try:
            r = self.smart_api.rmsLimit()
            return float(r['data']['availablecash']) if r['status'] else 0
        except BaseException:
            return 0

    def get_option_chain(self, symbol: str, expiry: str = "") -> List[Dict]:
        """Fetch options chain via Angel One SmartAPI."""
        if not self.is_authenticated or not self.smart_api:
            return []
        try:
            # Angel One option chain endpoint
            params = {"symbol": symbol, "expirydate": expiry} if expiry else {"symbol": symbol}
            r = self.smart_api.optionGreek(params)
            if not r or not r.get('status') or not r.get('data'):
                return []

            chain = []
            for item in r['data']:
                chain.append({
                    'strike': float(item.get('strikeprice', 0)),
                    'option_type': item.get('optiontype', ''),  # 'CE' or 'PE'
                    'expiry': item.get('expirydate', expiry),
                    'ltp': float(item.get('ltp', 0)),
                    'bid': float(item.get('bidprice', 0)),
                    'ask': float(item.get('askprice', 0)),
                    'oi': int(item.get('opninterest', 0)),
                    'oi_change': int(item.get('changeinopeninterest', 0)),
                    'volume': int(item.get('volume', 0)),
                    'iv': float(item.get('impliedvolatility', 0)),
                    'delta': float(item.get('delta', 0)),
                    'gamma': float(item.get('gamma', 0)),
                    'theta': float(item.get('theta', 0)),
                    'vega': float(item.get('vega', 0)),
                    'lot_size': int(item.get('lotsize', 1)),
                    'tradingsymbol': item.get('tradingsymbol', ''),
                })
            return chain
        except Exception as e:
            logger.error(f"AngelOne option chain error for {symbol}: {e}")
            return []


class UpstoxBroker(BaseBroker):
    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "Upstox"
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {}

    def login(self) -> bool:
        if 'access_token' in self.credentials:
            self.access_token = self.credentials['access_token']
            self.headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            self.is_authenticated = True
            return True
        return False

    def refresh_session(self) -> bool:
        """Refresh expired Upstox access token using refresh flow."""
        try:
            api_key = self.credentials.get('api_key', '')
            api_secret = self.credentials.get('api_secret', '')
            refresh_token = self.credentials.get('refresh_token', '')
            if not all([api_key, api_secret, refresh_token]):
                return False
            r = httpx.post(
                f"{self.base_url}/login/authorization/token",
                data={
                    'apiKey': api_key,
                    'apiSecret': api_secret,
                    'refreshToken': refresh_token,
                    'grant_type': 'refresh_token',
                },
                timeout=15,
            )
            data = r.json()
            if data.get('status') == 'success':
                self.credentials['access_token'] = data['data']['access_token']
                return self.login()
            return False
        except Exception:
            return False

    def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                r = httpx.get(url, headers=self.headers, params=data, timeout=15)
            elif method == "DELETE":
                r = httpx.delete(url, headers=self.headers, params=data, timeout=15)
            else:
                r = httpx.post(url, headers=self.headers, json=data, timeout=15)
            return r.json()
        except Exception:
            return {'status': 'error'}

    def place_order(self, order: Order) -> Order:
        if not self.is_authenticated:
            order.status = OrderStatus.REJECTED
            return order
        try:
            instrument_token = order.instrument_token
            if not instrument_token:
                if order.exchange == "NFO":
                    instrument_token = f"NSE_FO|{order.symbol}"
                else:
                    instrument_token = f"NSE_EQ|{order.symbol}"
            r = self._request("POST", "/order/place", {
                "quantity": order.quantity,
                "product": "D" if order.product == ProductType.CNC else "I",
                "validity": "DAY",
                "price": order.price,
                "instrument_token": instrument_token,
                "order_type": order.order_type.value,
                "transaction_type": order.transaction_type.value
            })
            if r.get('status') == 'success':
                order.order_id = r['data']['order_id']
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        r = self._request("DELETE", f"/order/cancel?order_id={order_id}")
        return r.get('status') == 'success'

    def get_order_status(self, order_id: str) -> OrderStatus:
        r = self._request("GET", f"/order/details?order_id={order_id}")
        if r.get('status') == 'success':
            return OrderStatus[r['data']['status'].upper()]
        return OrderStatus.PENDING

    def get_orders(self) -> List[Dict]:
        """Normalized order book via the Upstox order book endpoint."""
        try:
            r = self._request("GET", "/order/retrieve-all")
            if r.get('status') != 'success':
                return []
            raw = r.get('data', []) or []
            orders = []
            for o in raw:
                orders.append({
                    "order_id": str(o.get("order_id", "")),
                    "symbol": o.get("trading_symbol") or o.get("tradingsymbol", ""),
                    "transaction_type": str(o.get("transaction_type", "")).upper(),
                    "quantity": int(o.get("quantity", 0) or 0),
                    "filled_quantity": int(o.get("filled_quantity", 0) or 0),
                    "order_type": str(o.get("order_type", "")).upper(),
                    "price": float(o.get("price", 0) or 0),
                    "average_price": float(o.get("average_price", 0) or 0),
                    "status": str(o.get("status", "")).upper(),
                    "product": o.get("product", "") or "",
                })
            return orders
        except BaseException:
            return []

    def get_positions(self) -> List[Position]:
        positions = []
        r = self._request("GET", "/portfolio/short-term-positions")
        if r.get('status') == 'success':
            for p in r.get('data', []):
                if p['quantity'] != 0:
                    positions.append(Position(
                        symbol=p['trading_symbol'],
                        exchange=p['exchange'],
                        quantity=p['quantity'],
                        average_price=p['average_price'],
                        current_price=p['last_price'],
                        pnl=p['pnl'],
                        pnl_percent=0,
                        product=ProductType.CNC
                    ))
        return positions

    def get_holdings(self) -> List[Dict]:
        r = self._request("GET", "/portfolio/long-term-holdings")
        return r.get('data', []) if r.get('status') == 'success' else []

    def get_quote(self, symbol: str, exchange: str) -> Dict:
        r = self._request("GET", f"/market-quote/ltp?instrument_key=NSE_EQ|{symbol}")
        return r.get('data', {}) if r.get('status') == 'success' else {}

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        """
        Upstox doesn't support native GTT via API.
        Alternative: Place SL-M order for stop loss protection.
        Target is managed by the position monitor (scheduler).
        """
        if not self.is_authenticated or not gtt.trigger_values:
            gtt.status = "failed"
            return gtt
        try:
            sl_price = gtt.trigger_values[0]
            sell_order = gtt.orders[0] if gtt.orders else {}
            qty = sell_order.get('quantity', 0)
            if qty <= 0:
                gtt.status = "failed"
                return gtt
            instrument_token = f"NSE_EQ|{gtt.symbol}"
            r = self._request("POST", "/order/place", {
                "quantity": qty,
                "product": "D",
                "validity": "DAY",
                "price": 0,
                "trigger_price": sl_price,
                "instrument_token": instrument_token,
                "order_type": "SL-M",
                "transaction_type": sell_order.get('transaction_type', 'SELL'),
            })
            if r.get('status') == 'success':
                gtt.gtt_id = r['data'].get('order_id', '')
                gtt.status = "sl_placed"
                logger.info(f"Upstox SL-M order placed for {gtt.symbol} at {sl_price}")
            else:
                gtt.status = "sl_failed"
                logger.warning(f"Upstox SL-M failed: {r}")
        except Exception as e:
            gtt.status = "sl_failed"
            logger.error(f"Upstox GTT alternative error: {e}")
        return gtt

    def get_available_margin(self) -> float:
        r = self._request("GET", "/user/get-funds-and-margin")
        if r.get('status') == 'success':
            return float(r['data'].get('equity', {}).get('available_margin', 0))
        return 0

    def get_option_chain(self, symbol: str, expiry: str = "") -> List[Dict]:
        """Fetch options chain via Upstox v2 API."""
        if not self.is_authenticated:
            return []
        try:
            params = {"instrument_key": f"NSE_INDEX|{symbol}"}
            if expiry:
                params["expiry_date"] = expiry
            r = self._request("GET", "/option/chain", params)
            if r.get('status') != 'success' or not r.get('data'):
                return []

            chain = []
            for item in r['data']:
                for side in ('call_options', 'put_options'):
                    opt = item.get(side, {})
                    mkt = opt.get('market_data', {})
                    greeks = opt.get('option_greeks', {})
                    if not mkt:
                        continue
                    chain.append({
                        'strike': float(item.get('strike_price', 0)),
                        'option_type': 'CE' if side == 'call_options' else 'PE',
                        'expiry': item.get('expiry', expiry),
                        'ltp': float(mkt.get('ltp', 0)),
                        'bid': float(mkt.get('bid_price', 0)),
                        'ask': float(mkt.get('ask_price', 0)),
                        'oi': int(mkt.get('oi', 0)),
                        'oi_change': int(mkt.get('oi_day_change', 0)),
                        'volume': int(mkt.get('volume', 0)),
                        'iv': float(greeks.get('iv', 0)),
                        'delta': float(greeks.get('delta', 0)),
                        'gamma': float(greeks.get('gamma', 0)),
                        'theta': float(greeks.get('theta', 0)),
                        'vega': float(greeks.get('vega', 0)),
                    })
            return chain
        except Exception as e:
            logger.error(f"Upstox option chain error for {symbol}: {e}")
            return []


# ============================================================================
# FYERS (API v3) — OAuth2, symbol-based. BETA (2026-07-12): implemented to the
# documented Fyers API v3 surface; verify against a live account before real
# money. Every call is defensive — failures degrade to safe defaults so a wrong
# response never crashes the caller (it just broker-locks / rejects).
# ============================================================================

class FyersBroker(BaseBroker):
    BASE = "https://api-t1.fyers.in/api/v3"

    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "Fyers"
        self.app_id = credentials.get("app_id") or credentials.get("api_key") or ""
        self.access_token = credentials.get("access_token")

    def _headers(self) -> Dict:
        # Fyers auth header is "appId:accessToken".
        return {"Authorization": f"{self.app_id}:{self.access_token}"}

    def login(self) -> bool:
        if not self.app_id or not self.access_token:
            return False
        try:
            r = httpx.get(f"{self.BASE}/profile", headers=self._headers(), timeout=10)
            self.is_authenticated = r.status_code == 200 and r.json().get("s") == "ok"
            return self.is_authenticated
        except Exception as e:
            logger.warning("Fyers login failed: %s", e)
            return False

    def _fy_symbol(self, symbol: str, exchange: str) -> str:
        sym = symbol.upper().replace(".NS", "")
        if ":" in sym:
            return sym
        suffix = "-EQ" if exchange.upper() == "NSE" else ""
        return f"{exchange.upper()}:{sym}{suffix}"

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        try:
            fsym = self._fy_symbol(symbol, exchange)
            r = httpx.get(f"{self.BASE}/data/quotes", params={"symbols": fsym},
                          headers=self._headers(), timeout=10)
            if r.status_code != 200:
                return {}
            arr = (r.json() or {}).get("d") or []
            if not arr:
                return {}
            v = arr[0].get("v", {}) or {}
            # Normalize to the shape user_broker_data._normalize_quote understands.
            return {
                "last_price": v.get("lp"),
                "ohlc": {
                    "open": v.get("open_price"),
                    "high": v.get("high_price"),
                    "low": v.get("low_price"),
                    "close": v.get("prev_close_price"),
                },
                "volume": v.get("volume"),
                "net_change": v.get("ch"),
            }
        except Exception as e:
            logger.warning("Fyers get_quote(%s) failed: %s", symbol, e)
            return {}

    def get_historical(self, symbol: str, period: str = "1mo", interval: str = "1d"):
        """Return a list of {timestamp,open,high,low,close,volume} or None."""
        from datetime import date as _date, timedelta as _td
        res_map = {"1d": "D", "1day": "D", "1wk": "D", "1h": "60", "15m": "15", "5m": "5"}
        days = {"5d": 5, "1mo": 32, "3mo": 95, "6mo": 190, "1y": 370}.get(period, 32)
        try:
            fsym = self._fy_symbol(symbol, "NSE")
            to_d = _date.today()
            from_d = to_d - _td(days=days)
            r = httpx.get(
                f"{self.BASE}/data/history",
                params={
                    "symbol": fsym, "resolution": res_map.get(interval, "D"),
                    "date_format": "1", "range_from": from_d.isoformat(),
                    "range_to": to_d.isoformat(), "cont_flag": "1",
                },
                headers=self._headers(), timeout=15,
            )
            if r.status_code != 200:
                return None
            candles = (r.json() or {}).get("candles") or []
            out = []
            for c in candles:  # [epoch, o, h, l, c, v]
                if len(c) < 6:
                    continue
                from datetime import datetime as _dt, timezone as _tz
                out.append({
                    "timestamp": _dt.fromtimestamp(c[0], tz=_tz.utc).isoformat(),
                    "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
                    "close": float(c[4]), "volume": int(c[5]),
                })
            return out or None
        except Exception as e:
            logger.warning("Fyers get_historical(%s) failed: %s", symbol, e)
            return None

    def place_order(self, order: Order) -> Order:
        # Fyers order: type 1=Limit 2=Market 3=SL 4=SL-M; side 1=buy -1=sell.
        type_map = {OrderType.LIMIT: 1, OrderType.MARKET: 2, OrderType.SL: 3, OrderType.SL_M: 4}
        prod_map = {ProductType.CNC: "CNC", ProductType.MIS: "INTRADAY", ProductType.NRML: "MARGIN"}
        try:
            body = {
                "symbol": self._fy_symbol(order.symbol, order.exchange),
                "qty": order.quantity,
                "type": type_map.get(order.order_type, 2),
                "side": 1 if order.transaction_type == TransactionType.BUY else -1,
                "productType": prod_map.get(order.product, "CNC"),
                "limitPrice": order.price or 0,
                "stopPrice": order.trigger_price or 0,
                "validity": "DAY",
                "offlineOrder": False,
                "disclosedQty": 0,
            }
            r = httpx.post(f"{self.BASE}/orders/sync", json=body, headers=self._headers(), timeout=15)
            data = r.json() if r.status_code == 200 else {}
            if data.get("s") == "ok" and data.get("id"):
                order.order_id = str(data["id"])
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
                order.message = data.get("message", f"HTTP {r.status_code}")
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            r = httpx.delete(f"{self.BASE}/orders/sync", json={"id": order_id},
                             headers=self._headers(), timeout=10)
            return r.status_code == 200 and (r.json() or {}).get("s") == "ok"
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.PENDING

    def get_positions(self) -> List[Position]:
        try:
            r = httpx.get(f"{self.BASE}/positions", headers=self._headers(), timeout=10)
            rows = (r.json() or {}).get("netPositions") or [] if r.status_code == 200 else []
            out = []
            for p in rows:
                out.append(Position(
                    symbol=p.get("symbol", ""), exchange="NSE",
                    quantity=int(p.get("netQty", 0) or 0),
                    average_price=float(p.get("avgPrice", 0) or 0),
                    current_price=float(p.get("ltp", 0) or 0),
                    pnl=float(p.get("pl", 0) or 0), pnl_percent=0.0,
                    product=ProductType.CNC,
                ))
            return out
        except Exception:
            return []

    def get_holdings(self) -> List[Dict]:
        try:
            r = httpx.get(f"{self.BASE}/holdings", headers=self._headers(), timeout=10)
            return (r.json() or {}).get("holdings") or [] if r.status_code == 200 else []
        except Exception:
            return []

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        gtt.status = "not_supported"  # Fyers GTT not wired in BETA
        return gtt

    def get_available_margin(self) -> float:
        try:
            r = httpx.get(f"{self.BASE}/funds", headers=self._headers(), timeout=10)
            fund = (r.json() or {}).get("fund_limit") or [] if r.status_code == 200 else []
            for f in fund:
                if str(f.get("title", "")).lower().startswith("available"):
                    return float(f.get("equityAmount", 0) or 0)
            return 0.0
        except Exception:
            return 0.0


# ============================================================================
# DHAN (DhanHQ v2) — access-token (paste), REST. BETA (2026-07-12): connect +
# orders/positions/holdings/funds implemented to spec; get_quote needs Dhan's
# securityId master (not wired) so Dhan data broker-locks for now — connection
# and trading work. Verify against a live account before real money.
# ============================================================================

class DhanBroker(BaseBroker):
    BASE = "https://api.dhan.co/v2"

    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "Dhan"
        self.client_id = credentials.get("client_id") or credentials.get("dhan_client_id") or ""
        self.access_token = credentials.get("access_token")

    def _headers(self) -> Dict:
        return {
            "access-token": self.access_token or "",
            "client-id": self.client_id,
            "Content-Type": "application/json",
        }

    def login(self) -> bool:
        if not self.access_token or not self.client_id:
            return False
        try:
            r = httpx.get(f"{self.BASE}/fundlimit", headers=self._headers(), timeout=10)
            self.is_authenticated = r.status_code == 200
            return self.is_authenticated
        except Exception as e:
            logger.warning("Dhan login failed: %s", e)
            return False

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        # Dhan market feed keys on numeric securityId (needs the scrip master).
        # Not wired in BETA → return empty so the caller broker-locks honestly.
        return {}

    def place_order(self, order: Order) -> Order:
        # Dhan needs a numeric securityId (carried on order.instrument_token).
        seg = "NSE_EQ" if order.exchange.upper() == "NSE" else order.exchange.upper()
        prod_map = {ProductType.CNC: "CNC", ProductType.MIS: "INTRADAY", ProductType.NRML: "MARGIN"}
        otype_map = {OrderType.MARKET: "MARKET", OrderType.LIMIT: "LIMIT",
                     OrderType.SL: "STOP_LOSS", OrderType.SL_M: "STOP_LOSS_MARKET"}
        try:
            if not order.instrument_token:
                order.status = OrderStatus.REJECTED
                order.message = "Dhan requires a securityId (instrument_token)"
                return order
            body = {
                "dhanClientId": self.client_id,
                "transactionType": order.transaction_type.value,
                "exchangeSegment": seg,
                "productType": prod_map.get(order.product, "CNC"),
                "orderType": otype_map.get(order.order_type, "MARKET"),
                "securityId": str(order.instrument_token),
                "quantity": order.quantity,
                "price": order.price or 0,
                "triggerPrice": order.trigger_price or 0,
                "validity": "DAY",
            }
            r = httpx.post(f"{self.BASE}/orders", json=body, headers=self._headers(), timeout=15)
            data = r.json() if r.status_code in (200, 201) else {}
            if data.get("orderId"):
                order.order_id = str(data["orderId"])
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
                order.message = data.get("errorMessage") or f"HTTP {r.status_code}"
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            r = httpx.delete(f"{self.BASE}/orders/{order_id}", headers=self._headers(), timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.PENDING

    def get_positions(self) -> List[Position]:
        try:
            r = httpx.get(f"{self.BASE}/positions", headers=self._headers(), timeout=10)
            rows = r.json() if r.status_code == 200 else []
            out = []
            for p in rows or []:
                out.append(Position(
                    symbol=p.get("tradingSymbol", ""), exchange=p.get("exchangeSegment", "NSE"),
                    quantity=int(p.get("netQty", 0) or 0),
                    average_price=float(p.get("costPrice", 0) or 0),
                    current_price=float(p.get("ltp", 0) or 0),
                    pnl=float(p.get("realizedProfit", 0) or 0) + float(p.get("unrealizedProfit", 0) or 0),
                    pnl_percent=0.0, product=ProductType.CNC,
                ))
            return out
        except Exception:
            return []

    def get_holdings(self) -> List[Dict]:
        try:
            r = httpx.get(f"{self.BASE}/holdings", headers=self._headers(), timeout=10)
            return r.json() if r.status_code == 200 else []
        except Exception:
            return []

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        gtt.status = "not_supported"
        return gtt

    def get_available_margin(self) -> float:
        try:
            r = httpx.get(f"{self.BASE}/fundlimit", headers=self._headers(), timeout=10)
            data = r.json() if r.status_code == 200 else {}
            return float(data.get("availabelBalance", 0) or 0)
        except Exception:
            return 0.0


# ============================================================================
# KOTAK NEO — Bearer access token + "sid" session (pasted). BETA (2026-07-12):
# to the documented Neo API surface; quotes need Kotak's scrip master (not
# wired) so data broker-locks — connect + trading work. Verify with a live
# account before real money. Defensive throughout.
# ============================================================================

class KotakNeoBroker(BaseBroker):
    BASE = "https://gw-napi.kotaksecurities.com"

    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "Kotak Neo"
        self.client_id = credentials.get("client_id", "")
        self.access_token = credentials.get("access_token")
        self.sid = credentials.get("session_token", "")

    def _headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Sid": self.sid,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def login(self) -> bool:
        if not self.access_token:
            return False
        try:
            r = httpx.get(f"{self.BASE}/Orders/2.0/quick/user/positions",
                          headers=self._headers(), timeout=10)
            self.is_authenticated = r.status_code == 200
            return self.is_authenticated
        except Exception as e:
            logger.warning("Kotak Neo login failed: %s", e)
            return False

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        return {}  # scrip-token master not wired (BETA) → caller broker-locks

    def place_order(self, order: Order) -> Order:
        prod_map = {ProductType.CNC: "CNC", ProductType.MIS: "MIS", ProductType.NRML: "NRML"}
        try:
            body = {
                "am": "NO", "dq": "0", "es": order.exchange.lower() + "_cm",
                "mp": "0", "pc": prod_map.get(order.product, "CNC"),
                "pf": "N", "pr": str(order.price or 0), "pt": order.order_type.value,
                "qt": str(order.quantity), "rt": "DAY",
                "tp": str(order.trigger_price or 0), "ts": order.symbol,
                "tt": "B" if order.transaction_type == TransactionType.BUY else "S",
            }
            r = httpx.post(f"{self.BASE}/Orders/2.0/quick/order/rule/ms/place",
                           headers=self._headers(), json=body, timeout=15)
            data = r.json() if r.status_code == 200 else {}
            if data.get("nOrdNo") or data.get("orderId"):
                order.order_id = str(data.get("nOrdNo") or data.get("orderId"))
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
                order.message = data.get("errMsg") or f"HTTP {r.status_code}"
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            r = httpx.post(f"{self.BASE}/Orders/2.0/quick/order/cancel",
                           headers=self._headers(), json={"on": order_id}, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.PENDING

    def get_positions(self) -> List[Position]:
        try:
            r = httpx.get(f"{self.BASE}/Orders/2.0/quick/user/positions",
                          headers=self._headers(), timeout=10)
            rows = (r.json() or {}).get("data") or [] if r.status_code == 200 else []
            out = []
            for p in rows:
                out.append(Position(
                    symbol=p.get("trdSym", "") or p.get("sym", ""), exchange="NSE",
                    quantity=int(float(p.get("flBuyQty", 0) or 0) - float(p.get("flSellQty", 0) or 0)),
                    average_price=float(p.get("buyAmt", 0) or 0), current_price=float(p.get("ltp", 0) or 0),
                    pnl=float(p.get("posPnl", 0) or 0), pnl_percent=0.0, product=ProductType.CNC,
                ))
            return out
        except Exception:
            return []

    def get_holdings(self) -> List[Dict]:
        try:
            r = httpx.get(f"{self.BASE}/Portfolio/1.0/portfolio/v1/holdings",
                          headers=self._headers(), timeout=10)
            return (r.json() or {}).get("data") or [] if r.status_code == 200 else []
        except Exception:
            return []

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        gtt.status = "not_supported"
        return gtt

    def get_available_margin(self) -> float:
        try:
            r = httpx.post(f"{self.BASE}/Orders/2.0/quick/user/limits",
                           headers=self._headers(), json={"seg": "ALL", "exch": "ALL", "prod": "ALL"}, timeout=10)
            data = (r.json() or {}).get("data") or {} if r.status_code == 200 else {}
            return float(data.get("Net", 0) or 0)
        except Exception:
            return 0.0


# ============================================================================
# ALICE BLUE (ANT API) — API key + user session (pasted). BETA (2026-07-12).
# Quotes need Alice Blue's contract master (not wired) → data broker-locks;
# connect + trading work. Verify with a live account before real money.
# ============================================================================

class AliceBlueBroker(BaseBroker):
    BASE = "https://ant.aliceblueonline.com/rest/AliceBlueAPIService/api"

    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.name = "Alice Blue"
        self.user_id = credentials.get("client_id", "")
        self.access_token = credentials.get("access_token")

    def _headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.user_id} {self.access_token}",
            "Content-Type": "application/json",
        }

    def login(self) -> bool:
        if not self.access_token or not self.user_id:
            return False
        try:
            r = httpx.post(f"{self.BASE}/limits/getRmsLimits", headers=self._headers(), json={}, timeout=10)
            self.is_authenticated = r.status_code == 200
            return self.is_authenticated
        except Exception as e:
            logger.warning("Alice Blue login failed: %s", e)
            return False

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        return {}  # contract-master not wired (BETA) → caller broker-locks

    def place_order(self, order: Order) -> Order:
        prod_map = {ProductType.CNC: "CNC", ProductType.MIS: "MIS", ProductType.NRML: "NRML"}
        try:
            body = [{
                "complexty": "regular", "discqty": "0",
                "exch": order.exchange, "pCode": prod_map.get(order.product, "CNC"),
                "prctyp": order.order_type.value, "price": str(order.price or 0),
                "qty": str(order.quantity), "ret": "DAY",
                "trading_symbol": order.symbol, "transtype": order.transaction_type.value,
                "trigPrice": str(order.trigger_price or 0),
                "orderTag": "quantx",
            }]
            r = httpx.post(f"{self.BASE}/placeOrder/executePlaceOrder",
                           headers=self._headers(), json=body, timeout=15)
            data = r.json() if r.status_code == 200 else []
            first = data[0] if isinstance(data, list) and data else (data or {})
            if isinstance(first, dict) and (first.get("NOrdNo") or first.get("stat") == "Ok"):
                order.order_id = str(first.get("NOrdNo", ""))
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
                order.message = (first.get("emsg") if isinstance(first, dict) else None) or f"HTTP {r.status_code}"
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            r = httpx.post(f"{self.BASE}/placeOrder/cancelOrder",
                           headers=self._headers(), json={"nestOrderNumber": order_id}, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.PENDING

    def get_positions(self) -> List[Position]:
        try:
            r = httpx.post(f"{self.BASE}/positionAndHoldings/positionBook",
                           headers=self._headers(), json={"ret": "NET"}, timeout=10)
            rows = r.json() if r.status_code == 200 else []
            out = []
            for p in rows if isinstance(rows, list) else []:
                out.append(Position(
                    symbol=p.get("Tsym", ""), exchange=p.get("Exchange", "NSE"),
                    quantity=int(float(p.get("Netqty", 0) or 0)),
                    average_price=float(p.get("NetBuyavgprc", 0) or 0),
                    current_price=float(p.get("LTP", 0) or 0),
                    pnl=float(p.get("realisedprofitloss", 0) or 0), pnl_percent=0.0,
                    product=ProductType.CNC,
                ))
            return out
        except Exception:
            return []

    def get_holdings(self) -> List[Dict]:
        try:
            r = httpx.post(f"{self.BASE}/positionAndHoldings/holdings", headers=self._headers(), json={}, timeout=10)
            data = r.json() if r.status_code == 200 else {}
            return data.get("HoldingVal", []) if isinstance(data, dict) else []
        except Exception:
            return []

    def place_gtt_order(self, gtt: GTTOrder) -> GTTOrder:
        gtt.status = "not_supported"
        return gtt

    def get_available_margin(self) -> float:
        try:
            r = httpx.post(f"{self.BASE}/limits/getRmsLimits", headers=self._headers(), json={}, timeout=10)
            data = r.json() if r.status_code == 200 else []
            first = data[0] if isinstance(data, list) and data else {}
            return float(first.get("cashmarginavailable", 0) or 0)
        except Exception:
            return 0.0


class BrokerFactory:
    @staticmethod
    def create(broker_name: str, credentials: Dict) -> BaseBroker:
        brokers = {
            'zerodha': ZerodhaBroker,
            'angelone': AngelOneBroker,
            'upstox': UpstoxBroker,
            'fyers': FyersBroker,
            'dhan': DhanBroker,
            'kotakneo': KotakNeoBroker,
            'aliceblue': AliceBlueBroker,
        }
        if broker_name.lower() not in brokers:
            raise ValueError(f"Unknown broker: {broker_name}")
        return brokers[broker_name.lower()](credentials)


class TradeExecutor:
    def __init__(self, broker: BaseBroker):
        self.broker = broker

    def execute_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        target: float,
        capital: float,
        risk_percent: float = 3.0
    ) -> Dict:
        result = {'success': False, 'symbol': symbol, 'direction': direction}

        if confidence < 70:
            result['message'] = f"Confidence {confidence}% below 70%"
            return result

        risk_amount = capital * (risk_percent / 100)
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            result['message'] = "Invalid stop loss"
            return result

        quantity = int(risk_amount / risk_per_share)
        if quantity < 1:
            result['message'] = "Position size too small"
            return result

        margin = self.broker.get_available_margin()
        if quantity * entry_price > margin:
            quantity = int(margin / entry_price)
            if quantity < 1:
                result['message'] = "Insufficient margin"
                return result

        order = Order(
            symbol=symbol,
            exchange="NSE",
            transaction_type=TransactionType.BUY if direction == 'LONG' else TransactionType.SELL,
            quantity=quantity,
            product=ProductType.CNC if direction == 'LONG' else ProductType.NRML,
            order_type=OrderType.LIMIT,
            price=entry_price
        )

        order = self.broker.place_order(order)

        if order.status == OrderStatus.REJECTED:
            result['message'] = f"Order rejected: {order.message}"
            return result

        gtt = GTTOrder(
            symbol=symbol,
            exchange="NSE",
            trigger_type="two-leg",
            trigger_values=[stop_loss, target],
            orders=[
                {'transaction_type': 'SELL' if direction == 'LONG' else 'BUY', 'quantity': quantity, 'price': stop_loss},
                {'transaction_type': 'SELL' if direction == 'LONG' else 'BUY', 'quantity': quantity, 'price': target}
            ]
        )
        gtt = self.broker.place_gtt_order(gtt)

        result['success'] = True
        result['order_id'] = order.order_id
        result['gtt_id'] = gtt.gtt_id
        result['gtt_status'] = gtt.status
        result['quantity'] = quantity

        if gtt.status == "active":
            result['message'] = "Trade executed with GTT (SL + Target)"
        elif gtt.status == "sl_placed":
            result['message'] = "Trade executed with SL order (target managed by position monitor)"
        else:
            result['message'] = "Trade executed (SL/target managed by position monitor)"

        return result
