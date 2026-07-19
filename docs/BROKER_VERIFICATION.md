# Broker verification — BETA → stable

Four brokers ship **BETA** (implemented to their docs, not yet proven against a
live account): **Fyers, Dhan, Kotak Neo, Alice Blue**. To flip each to **stable**
we prove it once against a real account with `scripts/verify_broker.py`.

**You never share a token with anyone.** You run the script locally; it reads
your token through a hidden prompt and prints a pass/fail report that contains
**no secrets**. You send back the report — that's it.

---

## What I need from you (per broker you want to stabilise)

1. A funded (or at least logged-in) account with that broker.
2. A fresh API token for that broker (steps below — most expire daily).
3. The script output (paste it back).

That's all. I don't need your password, PIN, or token.

---

## Step 1 — get a token for the broker

### Fyers (OAuth)
1. Go to **myapi.fyers.in** → create an app → note the **App ID** (client_id) + **Secret**; set redirect `https://<your-domain>/broker/callback`.
2. Easiest for testing: connect Fyers once inside Quant X (Settings → Broker → Fyers) — that stores a valid access token. Or generate one from the Fyers API dashboard.
3. You'll enter: **app_id** and **access_token**.

### Dhan (token paste)
1. Open **web.dhan.co** → **Profile → DhanHQ Trading API → Access Token** → generate (valid ~30 days). Copy your **Client ID** too.
2. You'll enter: **client_id** and **access_token**.

### Kotak Neo (token paste)
1. Log in to the **Kotak Neo API portal** (napi.kotaksecurities.com), create an app, and generate your **access token** + **session id (sid)**.
2. You'll enter: **client_id** (UCC), **access_token**, **session_token** (the sid).

### Alice Blue (token paste)
1. Log in to **Alice Blue → Apps → API**, get your API key and generate a **session/access token**.
2. You'll enter: **client_id** (your User ID) and **access_token**.

---

## Step 2 — run the verifier (from the repo root)

```bash
# read-only checks (safe — no orders placed):
PYTHONPATH=. python scripts/verify_broker.py --broker fyers
PYTHONPATH=. python scripts/verify_broker.py --broker dhan
PYTHONPATH=. python scripts/verify_broker.py --broker kotakneo
PYTHONPATH=. python scripts/verify_broker.py --broker aliceblue
```

It asks for the fields above (secrets are typed hidden), then prints, e.g.:

```
=== Verifying fyers ... ===
  [✓] login
  [✓] get_quote               keys=['last_price','ohlc','volume','net_change']
  [✓] get_historical          22 candles
  [✓] get_positions           0 rows
  [✓] get_holdings            3 rows
  [✓] get_available_margin    48250.0
=== fyers: 6/6 checks passed ===
```

### Optional — prove the ORDER path (real, but unfillable)
Only when you're ready. This places a **₹1 limit buy of 1 share** (never fills at
that price) and **cancels it immediately** — it just proves order placement works:

```bash
PYTHONPATH=. python scripts/verify_broker.py --broker dhan --place-test-order
```

If the auto-cancel ever fails, the script tells you to cancel it manually in your
broker app. (It can't fill — the price is ₹1.)

---

## Step 3 — send me the output

Paste the whole `=== ... ===` block for each broker. No tokens appear in it.

---

## What each result means → what I do

| Result | Meaning | Fix |
|---|---|---|
| `login ✗` (token is fresh) | endpoint/field name off | I correct the login call |
| `get_quote ✗ empty` | needs the broker's instrument/scrip master | I wire symbol→token for that broker |
| `get_positions/holdings/margin ✗` | response shape differs | I adjust the field mapping |
| all `✓` | **verified** | I remove the BETA chip → **stable** |

Each broker that comes back all-green (login + reads, ideally + the order test)
I flip from BETA to stable in the same session you send the report.

---

## Per-broker "stable" bar

- **login** ✓ (mandatory)
- **get_positions / get_holdings / get_available_margin** ✓ (reads work)
- **get_quote** ✓ where applicable (Fyers today; Dhan/Kotak/Alice need their
  scrip master wired — I'll do that as part of stabilising, or we mark data as
  "via broker not yet available" and keep trading-only)
- **place_order + cancel_order** ✓ (the optional test order) — required before
  the broker is allowed to route real orders
