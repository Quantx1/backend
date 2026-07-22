"""
LLM adapter — single access surface for every agent. OpenRouter-only.

Every agent routes to an open-weight model via OpenRouter
(OpenAI-compatible, httpx). An agent's model comes from
``AGENT_MODEL_MAP`` (per-role), falling back to
``LLM_DEFAULT_MODEL``. When ``OPENROUTER_API_KEY`` is unset the adapter is
disabled and degrades gracefully (empty text / empty JSON / no stream) so dev
+ tests never hit the network.

"Mostly free" is the intent:
  • free→free→paid fallback — a ``:free`` slug that rate-limits (429)
    transparently rolls to the next free model, then a cheap paid one (last).
  • real SSE streaming via the gateway.

Every PAID call is gated by the monthly budget kill-switch (``llm_budget``):
once month-to-date spend crosses ``LLM_MONTHLY_BUDGET_USD`` ($50), paid calls
degrade gracefully and the free→paid spill is disabled (free still runs).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from ...core.config import settings
from ...observability.llm_budget import BudgetExceededError, get_meter
from ...observability.llm_pricing import is_paid, micros_usd_for

logger = logging.getLogger(__name__)

_BUDGET_MSG = (
    "The monthly AI budget has been reached, so this response is paused. "
    "AI features resume next month or once the budget is raised."
)
_PROVIDER = "openrouter"


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort: parse a JSON object out of a model reply (handles prose +
    code fences around it). Returns {} on failure."""
    if not text:
        return {}
    s = text.strip()
    # strip ```json fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(s[start: end + 1])
        except Exception:  # noqa: BLE001
            return {}
    return {}


# NOTE: OpenRouter caps the `models` array at 3 total → primary + ≤2 fallbacks.
_FALLBACK_CHAIN: Dict[str, list] = {
    "meta-llama/llama-3.3-70b-instruct:free": [
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct",          # paid — last resort
    ],
    "openai/gpt-oss-120b:free": [
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openai/gpt-oss-120b",                         # paid
    ],
    "qwen/qwen3-coder:free": [
        "qwen/qwen3-coder-30b-a3b-instruct",           # paid
    ],
    "qwen/qwen3-235b-a22b-2507": [
        "qwen/qwen3-next-80b-a3b-instruct:free",   # free spill on 429
        "meta-llama/llama-3.3-70b-instruct:free",  # free last resort
    ],
    "deepseek/deepseek-v3.2": [
        "deepseek/deepseek-v4-flash",              # cheaper paid
        "qwen/qwen3-next-80b-a3b-instruct:free",   # free last resort
    ],
}
_MAX_MODELS = 3


def build_models(model: str, *, allow_paid: bool) -> Optional[list]:
    """Ordered OpenRouter fallback list (free alts first, paid last). Returns
    None when no fallback applies. When over budget, paid entries are stripped."""
    chain = _FALLBACK_CHAIN.get(model)
    if chain is None:
        chain = [model[: -len(":free")]] if model.endswith(":free") else []
    candidates = [model] + list(chain)
    if not allow_paid:
        candidates = [c for c in candidates if c.endswith(":free")]
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    out = out[:_MAX_MODELS]
    return out if len(out) > 1 else None


# ── per-agent model map (AGENT_MODEL_MAP env JSON) ────────────────────
_MODEL_MAP_CACHE: Optional[Dict[str, str]] = None


def agent_model_map() -> Dict[str, str]:
    global _MODEL_MAP_CACHE
    if _MODEL_MAP_CACHE is None:
        raw = (settings.AGENT_MODEL_MAP or "").strip()
        try:
            _MODEL_MAP_CACHE = json.loads(raw) if raw else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("AGENT_MODEL_MAP is not valid JSON — ignoring: %s", exc)
            _MODEL_MAP_CACHE = {}
    return _MODEL_MAP_CACHE


# ── role → model tier (AIL v2). Env AGENT_MODEL_MAP overrides any of these.
# Chat-critical roles → fast paid Llama-70B (ultra-quick copilot); checked first.
_CHAT_FAST_ROLES = {"responder", "classifier", "tool_planner"}
_FAST_ROLES = {"sentiment"}
_STRONG_ROLES = {
    "grounded_reason", "scanner_thesis",
    "doctor", "debate", "strategy_gen", "fno_advisor",
}
_DEEP_ELIGIBLE_ROLES = {"doctor", "debate"}

# Market-brief roles run on the DEEP reasoning model unconditionally: these
# are ONE day-cached call each (briefing narrative + the what's-happening
# analysis), shared by every visitor — flagship public output where reasoning
# quality matters most and the marginal cost is a rounding error. The deep
# model's fallback chain still degrades gracefully when it's unavailable.
_MARKET_BRIEF_ROLES = {"market_brief"}


def _default_model_for(role: str) -> str:
    if role in _MARKET_BRIEF_ROLES:
        return settings.LLM_DEEP_MODEL
    if role in _CHAT_FAST_ROLES:
        return settings.LLM_CHAT_MODEL
    if role in _FAST_ROLES:
        return settings.LLM_FAST_MODEL
    if role in _STRONG_ROLES:
        return settings.LLM_STRONG_MODEL
    return settings.LLM_DEFAULT_MODEL


def resolve_model(role: Optional[str], *, deep: bool = False, tier: Optional[str] = None) -> str:
    """Resolve the model id for an agent role. Precedence: AGENT_MODEL_MAP env
    override > in-code role default > LLM_DEFAULT_MODEL. Deep mode (R1-class)
    only applies to deep-eligible roles for Elite users when the flag is on."""
    role = role or ""
    is_elite_deep = (deep and settings.LLM_DEEP_MODE_ENABLED
                     and str(tier or "").lower() == "elite")
    # Deep-eligible reasoning roles (doctor/debate) escalate to the R1-class deep model.
    if is_elite_deep and role in _DEEP_ELIGIBLE_ROLES:
        return settings.LLM_DEEP_MODEL
    # Chat-critical roles are fast by default (ultra-quick copilot), but a deep
    # Elite turn escalates the user-facing responder to the strong model — the
    # user opted into higher quality and accepts the extra latency.
    if is_elite_deep and role in _CHAT_FAST_ROLES:
        return settings.LLM_STRONG_MODEL
    env = agent_model_map().get(role)
    return env or _default_model_for(role)


def llm_for(role: str, *, deep: bool = False, tier: Optional[str] = None) -> "LLM":
    """LLM for an agent role — its resolved model (env > role default > default)."""
    return LLM(model=resolve_model(role, deep=deep, tier=tier))


def complete_sync(prompt: str, *, role: Optional[str] = None, model: Optional[str] = None,
                  system: Optional[str] = None, temperature: float = 0.2, top_p: float = 0.9,
                  feature: str = "agent", user_id: Optional[str] = None) -> str:
    """Synchronous OpenRouter completion for sync call-sites (e.g. scanner
    thesis). Returns '' when disabled / over budget / on error — never raises.
    Budget-gated + free→paid fallback."""
    import httpx
    if not settings.OPENROUTER_API_KEY:
        return ""
    mdl = model or resolve_model(role)
    meter = get_meter()
    over = meter.over_budget(settings.LLM_MONTHLY_BUDGET_USD)
    if is_paid(_PROVIDER, mdl) and over:
        return ""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload: Dict[str, Any] = {"model": mdl, "messages": messages, "temperature": temperature, "top_p": top_p}
    models = build_models(mdl, allow_paid=not over)
    if models:
        payload["models"] = models
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}", "Content-Type": "application/json",
        "HTTP-Referer": "https://quantx.app", "X-Title": "Quant X",
    }
    try:
        r = httpx.post(f"{settings.LLM_GATEWAY_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60.0)
        if r.status_code >= 400:
            logger.warning("sync gateway %s on %s: %s", r.status_code, mdl, r.text[:200])
            return ""
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("sync gateway call failed: %s", exc)
        return ""
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    used = data.get("model") or mdl
    usage = data.get("usage") or {}
    in_tok, out_tok = int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
    try:
        from ...observability import track_llm_usage
        track_llm_usage(
            user_id=user_id,
            feature=feature,
            provider=_PROVIDER,
            model=used,
            input_tokens=in_tok,
            output_tokens=out_tok)
        get_meter().record_micros(micros_usd_for(_PROVIDER, used, in_tok, out_tok))
    except Exception:  # noqa: BLE001
        pass
    return text.strip()


class LLM:
    def __init__(self, model: Optional[str] = None):
        self._model = model or settings.LLM_DEFAULT_MODEL

    @property
    def enabled(self) -> bool:
        return bool(settings.OPENROUTER_API_KEY)

    # ── budget ────────────────────────────────────────────────────────
    @staticmethod
    def _sb():
        try:
            from ...api.app import get_supabase_admin
            return get_supabase_admin()
        except Exception:  # noqa: BLE001
            return None

    def _over_budget(self) -> bool:
        meter = get_meter()
        try:
            meter.maybe_refresh(self._sb())
        except Exception:  # noqa: BLE001
            pass
        return meter.over_budget(settings.LLM_MONTHLY_BUDGET_USD)

    def _guard_budget(self, model: str) -> None:
        if is_paid(_PROVIDER, model) and self._over_budget():
            raise BudgetExceededError(
                f"Monthly LLM budget ${settings.LLM_MONTHLY_BUDGET_USD:.2f} exhausted.",
            )

    def _record(self, model, in_tok, out_tok, user_id, feature, metadata) -> None:
        try:
            from ...observability import track_llm_usage
            track_llm_usage(
                user_id=user_id, feature=feature, provider=_PROVIDER, model=model,
                input_tokens=in_tok, output_tokens=out_tok, metadata=metadata or {},
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            get_meter().record_micros(micros_usd_for(_PROVIDER, model, in_tok, out_tok))
        except Exception:  # noqa: BLE001
            pass

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://quantx.app",
            "X-Title": "Quant X",
        }

    def _payload(self, model, prompt, system, temperature, top_p, *                 , allow_paid_fallback, response_json=False, stream=False) -> Dict[str, Any]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt},
        ]
        payload: Dict[str, Any] = {
            "model": model, "messages": messages,
            "temperature": temperature, "top_p": top_p,
        }
        models = build_models(model, allow_paid=allow_paid_fallback)
        if models:
            payload["models"] = models
        # Route to the fastest provider (Groq/Cerebras/SambaNova/…) for ultra-quick
        # responses instead of OpenRouter's default price-balanced routing (which
        # occasionally lands a slow host — the ~20s outliers). require_parameters
        # keeps JSON-mode (planner) on providers that actually support it.
        payload["provider"] = {"sort": "throughput", "require_parameters": True}
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def _gateway_chat(self, model, prompt, system, temperature, top_p, *                            , allow_paid_fallback, response_json=False) -> Tuple[str, int, int, str]:
        import httpx
        payload = self._payload(model, prompt, system, temperature, top_p,
                                allow_paid_fallback=allow_paid_fallback, response_json=response_json)
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{settings.LLM_GATEWAY_BASE_URL}/chat/completions",
                headers=self._headers(), json=payload,
            )
            if resp.status_code >= 400:
                logger.warning("gateway %s on %s: %s", resp.status_code, model, resp.text[:300])
            resp.raise_for_status()
            data = resp.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        used = data.get("model") or model
        return text, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0), used

    async def _gateway_stream(self, model, prompt, system, temperature, top_p, *                              , allow_paid_fallback, usage_out: Dict[str, Any]) -> AsyncIterator[str]:
        import httpx
        payload = self._payload(model, prompt, system, temperature, top_p,
                                allow_paid_fallback=allow_paid_fallback, stream=True)
        used, in_tok, out_tok = model, 0, 0
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{settings.LLM_GATEWAY_BASE_URL}/chat/completions",
                headers=self._headers(), json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    logger.warning("gateway stream %s on %s: %s", resp.status_code, model, body[:300])
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except Exception:  # noqa: BLE001
                        continue
                    used = chunk.get("model") or used
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content")
                        if delta:
                            yield delta
                    u = chunk.get("usage")
                    if u:
                        in_tok = int(u.get("prompt_tokens", 0) or 0)
                        out_tok = int(u.get("completion_tokens", 0) or 0)
        usage_out["model"], usage_out["in"], usage_out["out"] = used, in_tok, out_tok

    # ── public API ────────────────────────────────────────────────────
    async def complete(self, prompt: str, *, temperature: float = 0.2, top_p: float = 0.9,
                       system: Optional[str] = None, user_id: Optional[str] = None,
                       feature: str = "agent", metadata: Optional[Dict[str, Any]] = None) -> str:
        if not self.enabled:
            return ""
        model = self._model
        try:
            self._guard_budget(model)
        except BudgetExceededError:
            return _BUDGET_MSG
        allow = not self._over_budget()
        text, in_tok, out_tok, used = await self._gateway_chat(
            model, prompt, system, temperature, top_p, allow_paid_fallback=allow)
        self._record(used, in_tok, out_tok, user_id, feature, metadata)
        return text

    async def complete_stream(self, prompt: str, *, temperature: float = 0.2, top_p: float = 0.9,
                              system: Optional[str] = None, user_id: Optional[str] = None,
                              feature: str = "agent", metadata: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        if not self.enabled:
            return
        model = self._model
        try:
            self._guard_budget(model)
        except BudgetExceededError:
            yield _BUDGET_MSG
            return
        allow = not self._over_budget()
        usage: Dict[str, Any] = {}
        try:
            async for piece in self._gateway_stream(
                    model, prompt, system, temperature, top_p,
                    allow_paid_fallback=allow, usage_out=usage):
                yield piece
        finally:
            if usage:
                self._record(usage.get("model", model), usage.get("in", 0),
                             usage.get("out", 0), user_id, feature, metadata)

    async def generate_json(self, prompt: str, schema_hint: str, *, temperature: float = 0.0,
                            system: Optional[str] = None, user_id: Optional[str] = None,
                            feature: str = "agent", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if system is None:
            system = "Respond with a single JSON object only. No prose."
        full = f"{system}\n\nExpected JSON shape:\n{schema_hint}\n\nTask:\n{prompt}"
        model = self._model
        try:
            self._guard_budget(model)
        except BudgetExceededError:
            return {}
        allow = not self._over_budget()
        raw, in_tok, out_tok, used = await self._gateway_chat(
            model, full, None, temperature, 0.1, allow_paid_fallback=allow, response_json=True)
        self._record(used, in_tok, out_tok, user_id, feature, metadata)
        return extract_json(raw)

    async def complete_vision(self, prompt: str, image_b64: str, *, mime: str = "image/png",
                              temperature: float = 0.2, top_p: float = 0.9, user_id: Optional[str] = None,
                              feature: str = "vision", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Multimodal: text prompt + one base64 image. Requires a vision-capable
        model (set via AGENT_MODEL_MAP['vision']). Returns '' on any failure."""
        if not self.enabled:
            return ""
        model = self._model
        try:
            self._guard_budget(model)
        except BudgetExceededError:
            return ""
        allow = not self._over_budget()
        import httpx
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]
        payload: Dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}],
                                   "temperature": temperature, "top_p": top_p}
        models = build_models(model, allow_paid=allow)
        if models:
            payload["models"] = models
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{settings.LLM_GATEWAY_BASE_URL}/chat/completions",
                    headers=self._headers(), json=payload)
                if resp.status_code >= 400:
                    logger.warning("vision gateway %s on %s: %s", resp.status_code, model, resp.text[:200])
                    return ""
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("vision gateway call failed: %s", exc)
            return ""
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        used = data.get("model") or model
        self._record(used, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0),
                     user_id, feature, metadata)
        return text


# ---------------------------------------------------------------- singleton
_llm: Optional[LLM] = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
