# Setup checklist

1. Create a new Hugging Face Space.
2. Choose **Gradio**.
3. Set visibility to **Private**.
4. Upload all files from this ZIP into the root of the Space.
5. Add Secrets:
   - `GEMINI_API_KEY`
   - `BINANCE_LIVE_API_KEY`
   - `BINANCE_LIVE_API_SECRET`
   - `SFM_TRADING_LIVE_ACK=I_UNDERSTAND_LIVE_TRADING_CAN_LOSE_MONEY`
   - `SFM_TRADING_OPERATIVE_ACK=I_ACCEPT_HIGHER_AUTONOMY_RISK`
6. Factory rebuild.
7. First run: paper mode only.
8. Second run: live mode with one cycle and small limits.

Never commit API keys into files. Use Hugging Face Secrets only.
