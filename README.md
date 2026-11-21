# Cognitively Aware AI Nutrition Chatbot

A course project using Python, Gradio, and Anthropic Claude to combine a deterministic nutrition rule engine with LLM-generated recommendations and contextual follow-up chat.

## Features
- Exact and RapidFuzz-based matching across **107 food records**, followed by Claude estimation for unmatched items when an API key is configured.
- Per-serving nutrient calculations and heuristic dietary flags.
- Conversational follow-up questions and **per-browser-session** memory (not persistent storage).
- Prompting based on four behavioral frameworks: Cognitive Load Theory, Health Belief Model, Nudge Theory, and Information Foraging Theory.
- Plotly macronutrient and recent-analysis trend charts (the last seven analyses, **not** necessarily seven calendar days).
- Pydantic validation of Claude-generated nutrient estimates.

## Run locally

```bash
python -m venv .venv
# Activate the virtual environment (Windows: .venv\Scripts\activate; macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` as an environment variable or in a local `.env` loader of your choice. **Never commit your key.** The app still performs rule-based analysis without one.

```bash
python app.py
```

Then open http://localhost:7860.

## Files
- `app.py`: Gradio user interface and event handlers
- `nutrition_engine.py`: food matching, rules, memory class, Claude API integration
- `data/foods.csv`: built-in sample nutritional records
- `tests/`: deterministic tests
- `notebooks/IS596_FINAL_sanitized.ipynb`: original Colab notebook with notebook outputs cleared and the embedded API credential replaced. **Historical notebook code has not been refactored** to match `app.py`.

## Disclaimer
This is an educational demo, not medical or individualized dietary advice. Nutrient records are approximate, LLM fallback estimates are unverified, and the rule thresholds are educational heuristics. Do not enter sensitive health information.

## Demo
Add your recorded demo video URL here after reviewing it for privacy.
