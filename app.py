import gradio as gr
import os
import anthropic
from nutrition_engine import (SessionTracker, ANTHROPIC_API_KEY, run_rule_engine,
    call_llm_analysis, call_llm_followup, generate_weekly_summary, parse_llm_sections)
import plotly.graph_objects as go
import plotly.express as px

def make_macro_chart(totals: dict) -> go.Figure:
    """Donut chart showing macronutrient breakdown by calories."""
    protein_cal = totals["protein"] * 4
    carbs_cal   = totals["carbs"]   * 4
    fats_cal    = totals["fats"]    * 9
    total_mac   = protein_cal + carbs_cal + fats_cal
    if total_mac == 0:
        return go.Figure()
    fig = go.Figure(go.Pie(
        labels=["Protein","Carbohydrates","Fats"],
        values=[protein_cal, carbs_cal, fats_cal],
        hole=0.45,
        marker_colors=["#4CAF50","#FF9800","#2196F3"],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.0f} kcal (%{percent})<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=f"Macro Split — {round(totals['calories'])} kcal total",
                   font=dict(size=14)),
        margin=dict(t=50,b=10,l=10,r=10),
        height=280,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def make_trend_chart(session: SessionTracker) -> go.Figure | None:
    """Line chart of daily calories over session history."""
    if len(session.daily_logs) < 2:
        return None
    days    = [f"Day {i+1}" for i in range(len(session.daily_logs))]
    cals    = [d["calories"]  for d in session.daily_logs]
    proteins= [d["protein"]   for d in session.daily_logs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=cals,     mode="lines+markers",
                             name="Calories",  line=dict(color="#FF9800",width=2)))
    fig.add_trace(go.Scatter(x=days, y=proteins, mode="lines+markers",
                             name="Protein (g)",line=dict(color="#4CAF50",width=2)))
    fig.add_hline(y=2000, line_dash="dash", line_color="gray",
                  annotation_text="2000 kcal target")
    fig.update_layout(
        title="Nutritional Trends Across Sessions",
        xaxis_title="Session", yaxis_title="Amount",
        height=260, margin=dict(t=50,b=30,l=40,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=-0.25)
    )
    return fig

# ── MAIN ANALYSIS HANDLER ──────────────────────────────────────────────────
def analyze(diet_pref, goal,
            b_food, b_srv, l_food, l_srv, d_food, d_srv,
            snack_food, snack_srv, history, session):

    key = ANTHROPIC_API_KEY

    session = session or SessionTracker()
    history = list(history or [])
    meals_input = [
        ("Breakfast", b_food,     b_srv),
        ("Lunch",     l_food,     l_srv),
        ("Dinner",    d_food,     d_srv),
        ("Snack",     snack_food, snack_srv),
    ]

    facts = run_rule_engine(diet_pref, goal, meals_input, key)

    if facts is None:
        msg = "Please enter at least one food item to begin your analysis."
        history.append({"role":"assistant","content":msg})
        return history, go.Figure(), None, "", "", "", "", session

    # Log this session
    session.add_day(facts)
    weekly = session.get_weekly_trends()

    # Macro chart
    macro_fig = make_macro_chart(facts["totals"])
    trend_fig = make_trend_chart(session)

    if not key:
        # Fallback: rule-based output without LLM
        t = facts["totals"]
        summary = (f"### 🍽 Nutrition Summary\n"
                   + "\n".join(f"- {l}" for l in facts["meal_lines"])
                   + f"\n\n**Total:** {t['calories']} kcal | Protein: {t['protein']}g | "
                   f"Carbs: {t['carbs']}g | Fiber: {t['fiber']}g | Sugar: {t['sugar']}g | Sodium: {t['sodium_mg']}mg\n"
                   f"**Overall Status: {facts['status']}**\n\n_⚠️ Add API key for LLM-powered guidance._")
        issues = "### ⚠️ Issues Detected\n" + ("\n".join(f"- **{i[0]}** — {i[1]}" for i in facts["issues"]) or "- None")
        recs   = "### ✅ Recommendations\n- Add your Anthropic API key for personalized LLM recommendations."
        nudge  = "### 💡 Nudge for Tomorrow\n- Add your API key for a personalized nudge."
        response_text = f"{summary}\n\n{issues}\n\n{recs}\n\n{nudge}"
    else:
        try:
            response_text = call_llm_analysis(facts, key, weekly, session)
        except anthropic.AuthenticationError:
            response_text = "❌ Invalid API key. Please check and re-enter."
        except Exception as e:
            response_text = f"❌ API error: {e}"

    # Update session memory
    meal_summary = "\n".join(facts["meal_lines"])
    session.conversation_history.append({"role":"user",      "content":f"Meal analysis request:\n{meal_summary}"})
    session.conversation_history.append({"role":"assistant", "content":response_text})


    session.conversation_history = session.conversation_history[-10:]

    summary, issues, recs, nudge = parse_llm_sections(response_text)
    if facts["match_notes"]:
        nudge += "\n\n_ℹ️ Lookup notes: " + "; ".join(facts["match_notes"]) + "_"

    history.append({"role":"user",      "content":f"Analyzed: {', '.join(l.split(':')[1].split('[')[0].strip() for l in facts['meal_lines'])}"})
    history.append({"role":"assistant", "content":response_text})

    return history, macro_fig, trend_fig, summary, issues, recs, nudge, session

# ── FOLLOW-UP HANDLER ──────────────────────────────────────────────────────
def followup(user_question, history, session):
    session = session or SessionTracker()
    history = list(history or [])
    key = ANTHROPIC_API_KEY
    if not user_question.strip():
        return history, session, ""
    if not key:
        history.append({"role":"user",      "content":user_question})
        history.append({"role":"assistant", "content":"Please add your Anthropic API key to use conversational follow-up."})
        return history, session, ""
    try:
        response = call_llm_followup(user_question, key, session)
        session.conversation_history.append({"role":"user",      "content":user_question})
        session.conversation_history.append({"role":"assistant", "content":response})
        session.conversation_history = session.conversation_history[-10:]
    except Exception as e:
        response = f"❌ Error: {e}"
    history.append({"role":"user",      "content":user_question})
    history.append({"role":"assistant", "content":response})
    return history, session, ""

# ── WEEKLY SUMMARY HANDLER ─────────────────────────────────────────────────
def get_weekly_summary(session):
    session = session or SessionTracker()
    if not ANTHROPIC_API_KEY:
        return "Set ANTHROPIC_API_KEY in the server environment to generate an AI summary."
    return generate_weekly_summary(ANTHROPIC_API_KEY, session)

def reset_all():
    return [], go.Figure(), None, "", "", "", "", "", "", SessionTracker()

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(theme=gr.themes.Soft(), title="Cognitively Aware Nutrition Assistant") as demo:

    gr.Markdown("""
# 🥗 Cognitively Aware AI Nutrition Decision Assistant
**IS596 — Spring 2026** | Grounded in Cognitive Load Theory · Health Belief Model · Behavioral Nudge Theory · Information Foraging Theory

_Type any food — Indian or international. The system matches it automatically, or asks Claude to estimate if it's not in the database._
    """)

    with gr.Row():
        # ── LEFT COLUMN: INPUT ────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            gr.Markdown("### 🔐 Model Connection")
            gr.Markdown("Claude is configured through a server-side API key. Without a key, the rule-based analysis still works.")

            gr.Markdown("### 👤 Your Profile")
            diet_input = gr.Dropdown(
                ["Vegetarian","Vegan","Non-Vegetarian","No Preference"],
                label="Diet Preference", value="Vegetarian"
            )
            goal_input = gr.Dropdown(
                ["Weight Loss","Weight Maintenance","Muscle Gain",
                 "Energy Improvement","Blood Sugar Control","Heart Health"],
                label="Health Goal", value="Weight Loss"
            )

            gr.Markdown("### 🍳 Meals Today")
            gr.Markdown("_Servings = how many standard portions. 2 rotis = 2 servings._")

            b_food = gr.Textbox(label="Breakfast food", placeholder="e.g. oats, idli, poha, toast")
            b_srv  = gr.Slider(0.5,5,value=1,step=0.5,label="Breakfast servings")

            l_food = gr.Textbox(label="Lunch food", placeholder="e.g. dal, roti, rice, chicken breast")
            l_srv  = gr.Slider(0.5,5,value=1,step=0.5,label="Lunch servings")

            d_food = gr.Textbox(label="Dinner food", placeholder="e.g. tofu, quinoa, sabzi, salmon")
            d_srv  = gr.Slider(0.5,5,value=1,step=0.5,label="Dinner servings")

            snack_food = gr.Textbox(label="Snack (optional)", placeholder="e.g. banana, almonds, greek yogurt")
            snack_srv  = gr.Slider(0.5,5,value=1,step=0.5,label="Snack servings")

            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                reset_btn   = gr.Button("🔄 Reset",   variant="secondary")

        # ── RIGHT COLUMN: OUTPUT ──────────────────────────────────────────
        with gr.Column(scale=2):

            with gr.Tabs():

                with gr.Tab("📊 Today's Analysis"):
                    macro_chart   = gr.Plot(label="Macro breakdown")
                    summary_out   = gr.Markdown()
                    issues_out    = gr.Markdown()
                    recs_out      = gr.Markdown()
                    nudge_out     = gr.Markdown()

                with gr.Tab("💬 Conversation"):
                    gr.Markdown("_Ask follow-up questions about your nutrition. The chatbot remembers this session's context._")
                    chatbot = gr.Chatbot(label="Nutrition Chatbot", height=400)
                    with gr.Row():
                        followup_input = gr.Textbox(
                            label="Your question",
                            placeholder="e.g. What if I swapped rice for quinoa? Can you suggest a high-protein breakfast?",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Tab("📈 Weekly Trends"):
                    gr.Markdown("_Tracks nutrition across multiple sessions. Run at least 2 analyses to see trends._")
                    trend_chart   = gr.Plot(label="Nutritional trends")
                    weekly_btn    = gr.Button("📝 Generate Weekly Summary", variant="secondary")
                    weekly_out    = gr.Markdown()

    gr.Markdown("""
---
**Serving size reference** — 1 serving means:
`Roti/Chapati` = 1 piece (40g) · `Rice` = 1 cup cooked (186g) · `Dal` = 1 bowl (240g) · `Oats` = 1 bowl dry (40g) · `Egg` = 1 large egg (50g)
`Banana` = 1 medium (118g) · `Chicken breast` = 1 fillet (174g) · `Greek yogurt` = 1 cup (245g) · `Almonds` = ~23 nuts (28g) · `Paneer` = 100g portion

> ⚕️ **Disclaimer** — This system provides general dietary guidance only and is not a substitute for advice from a registered dietitian or healthcare professional.
    """)

    session_state = gr.State(SessionTracker())

    # ── EVENT HANDLERS ────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze,
        inputs=[diet_input, goal_input,
                b_food, b_srv, l_food, l_srv, d_food, d_srv,
                snack_food, snack_srv, chatbot, session_state],
        outputs=[chatbot, macro_chart, trend_chart,
                 summary_out, issues_out, recs_out, nudge_out, session_state]
    )
    send_btn.click(
        fn=followup,
        inputs=[followup_input, chatbot, session_state],
        outputs=[chatbot, session_state, followup_input]
    )
    followup_input.submit(
        fn=followup,
        inputs=[followup_input, chatbot, session_state],
        outputs=[chatbot, session_state, followup_input]
    )
    weekly_btn.click(
        fn=get_weekly_summary,
        inputs=[session_state],
        outputs=[weekly_out]
    )
    reset_btn.click(
        fn=reset_all,
        outputs=[chatbot, macro_chart, trend_chart,
                 summary_out, issues_out, recs_out, nudge_out, weekly_out, session_state]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=False)