
# Cognitively Aware AI Nutrition Chatbot

An interactive GenAI nutrition assistant built with **Anthropic Claude** and **Gradio**. It combines nutritional analysis with personalized recommendations, multi-turn conversations, and meal-trend visualizations.

## Features

- **Conversational AI:** Context-aware follow-up questions with session-based chat memory.
- **Cognitively Aware Prompts:** Personalized recommendations guided by Cognitive Load Theory, the Health Belief Model, Nudge Theory, and Information Foraging Theory.
- **Hybrid Nutrition Analysis:** Rule-based nutrient calculations and dietary checks combined with Claude-generated insights.
- **Intelligent Food Matching:** Exact lookup and RapidFuzz similarity matching across **107 food records**, with Claude-based estimation for unmatched foods.
- **Visual Insights:** Interactive macronutrient charts and recent meal-analysis trends using Plotly.
- **Structured AI Output:** Pydantic validation for generated nutritional estimates.

## Tech Stack

**Python · Anthropic Claude API · Gradio · Pandas · RapidFuzz · Pydantic · Plotly**

## Demo

[Watch Project Demonstration](https://mediaspace.illinois.edu/media/t/1_st7dvq7y)

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your `ANTHROPIC_API_KEY` environment variable and run:

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

## Project Structure

```text
app.py                 # Gradio interface
nutrition_engine.py    # Food matching, nutrition rules, Claude integration
requirements.txt       # Python dependencies
data/foods.csv         # Nutrition dataset
tests/                 # Automated nutrition tests
notebooks/             # Colab notebook
```
