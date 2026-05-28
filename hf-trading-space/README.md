---
title: LangGraph SFM Trading Private
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.33.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Private LangGraph-SFM trading control panel with Gemini strategy planning and SFM safety gates.
tags:
  - langgraph
  - ai-agents
  - ai-safety
  - trading
  - gradio
  - gemini
---

# LangGraph-SFM Trading Private Space

Private Hugging Face Space for controlled paper/testnet/live experiments with:

- Gemini news + signal advisor
- LLM strategy planner
- SFM strategy critic and final gate
- hard risk policy
- Binance Spot execution adapter
- audit JSON output
- kill switch

## Keep this Space private

Do **not** make this Space public if Binance live keys are configured. A public Space could expose a live trading button to visitors.

## Required Secrets

Set these under **Space → Settings → Secrets**:

```text
GEMINI_API_KEY=...
BINANCE_LIVE_API_KEY=...
BINANCE_LIVE_API_SECRET=...
SFM_TRADING_LIVE_ACK=I_UNDERSTAND_LIVE_TRADING_CAN_LOSE_MONEY
SFM_TRADING_OPERATIVE_ACK=I_ACCEPT_HIGHER_AUTONOMY_RISK
```

Optional testnet secrets:

```text
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
SFM_TRADING_TESTNET_ACK=I_UNDERSTAND_TESTNET_ORDERS
```

Recommended non-secret Variables:

```text
SFM_TRADING_LIVE_PROFILE=operative
SFM_TRADING_GEMINI_MODEL=gemini-2.5-flash
```

## Safety defaults

The UI defaults to paper mode. Live execution requires:

1. mode = `live`
2. `enable real execution` checkbox
3. `allow live` checkbox
4. exact confirmation phrase in the UI
5. live ack secrets
6. Binance live API secrets

If SFM returns `review` or `block`, the app does not submit an order.
