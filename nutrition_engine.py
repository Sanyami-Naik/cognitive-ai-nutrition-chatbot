import os, json, datetime
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict
import anthropic
from rapidfuzz import process, fuzz

DATA_PATH = Path(__file__).resolve().parent / "data" / "foods.csv"
foods_df = pd.read_csv(DATA_PATH)
foods_df["food_name"] = foods_df["food_name"].str.lower().str.strip()
KNOWN_FOODS = foods_df["food_name"].tolist()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

class NutritionEstimate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cal_per_100g: float = Field(ge=0, le=1000)
    protein: float = Field(ge=0, le=100)
    carbs: float = Field(ge=0, le=100)
    fats: float = Field(ge=0, le=100)
    fiber: float = Field(ge=0, le=100)
    sugar: float = Field(ge=0, le=100)
    sodium_mg: float = Field(ge=0, le=50000)
    gi: float = Field(ge=0, le=100)
    default_serving_g: float = Field(gt=0, le=2000)
    serving_label: str
    category: str
    diet_tags: str

# ─── ALIASES: common alternate names → canonical CSV name ──────────────────
ALIASES = {
    "greek yoghurt":"greek yogurt", "chapatti":"chapati",
    "rajmah":"rajma", "dal rice":"dal", "curd":"dahi",
    "vegetable curry":"sabzi", "mixed veg":"mixed vegetables",
    "stir fry":"mixed vegetables", "sunny side up":"egg",
    "fried egg":"egg", "green salad":"salad",
    "brown bread":"whole wheat bread", "wheat bread":"whole wheat bread",
    "trail mix":"nuts", "peanuts":"nuts",
    "dal makhani":"dal", "chana":"chickpeas",
}

# ─── LAYER 1 + 2: CSV LOOKUP ───────────────────────────────────────────────
def lookup_food_csv(food_name: str) -> dict | None:
    name = food_name.strip().lower()
    name = ALIASES.get(name, name)
    # Exact match
    exact = foods_df[foods_df["food_name"] == name]
    if not exact.empty:
        r = exact.iloc[0].to_dict()
        r["match_type"] = "exact"
        return r
    # Fuzzy match
    hit = process.extractOne(name, KNOWN_FOODS, scorer=fuzz.WRatio, score_cutoff=72)
    if hit:
        matched, score, _ = hit
        r = foods_df[foods_df["food_name"] == matched].iloc[0].to_dict()
        r["match_type"] = f"fuzzy→{matched} ({score:.0f}%)"
        return r
    return None

# ─── LAYER 3: LLM NUTRITION ESTIMATE ─────────────────────────────────────
def estimate_food_llm(food_name: str, api_key: str) -> dict | None:
    if not api_key:
        return None
    prompt = f"""Estimate nutrition per 100g for: \"{food_name}\"
Also give: default_serving_g (grams in one typical serving) and serving_label (plain English).
ONLY return valid JSON, no markdown:
{{"cal_per_100g":0,"protein":0,"carbs":0,"fats":0,"fiber":0,"sugar":0,"sodium_mg":0,"gi":0,"default_serving_g":100,"serving_label":"1 serving (100g)","category":"healthy","diet_tags":"vegetarian"}}"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL, max_tokens=250,
            messages=[{"role":"user","content":prompt}]
        )
        text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        data = NutritionEstimate.model_validate(json.loads(text)).model_dump()
        data["food_name"] = food_name.lower()
        data["match_type"] = "llm_estimate"
        return data
    except Exception as e:
        print(f"  LLM estimate failed for '{food_name}': {e}")
        return None

def get_food(food_name: str, api_key: str) -> dict:
    r = lookup_food_csv(food_name)
    if r:
        return r
    print(f"  '{food_name}' not in CSV — asking Claude to estimate...")
    llm = estimate_food_llm(food_name, api_key)
    return llm or {
        "food_name":food_name.lower(),"cal_per_100g":200,"protein":5,
        "carbs":25,"fats":8,"fiber":2,"sugar":5,"sodium_mg":200,"gi":50,
        "default_serving_g":100,"serving_label":"1 serving (~100g, estimated)",
        "category":"unknown","diet_tags":"vegetarian","match_type":"placeholder"
    }

def compute_nutrients(row: dict, servings: float) -> dict:
    """Accurate per-serving calculation: nutrient × (servings × default_g) / 100"""
    g = float(row.get("default_serving_g", 100)) * servings
    f = g / 100.0
    return {
        "calories":  round(float(row.get("cal_per_100g",0)) * f, 1),
        "protein":   round(float(row.get("protein",0))      * f, 1),
        "carbs":     round(float(row.get("carbs",0))        * f, 1),
        "fats":      round(float(row.get("fats",0))         * f, 1),
        "fiber":     round(float(row.get("fiber",0))        * f, 1),
        "sugar":     round(float(row.get("sugar",0))        * f, 1),
        "sodium_mg": round(float(row.get("sodium_mg",0))   * f, 1),
    }

# ─── RULE ENGINE ──────────────────────────────────────────────────────────
def run_rule_engine(diet_pref, goal, meals_input, api_key):
    """
    meals_input: list of (meal_name, food_name, servings)
    Returns structured facts dict for the LLM.
    """
    entries = []
    for meal_name, food, srv in meals_input:
        if food and str(food).strip():
            try: srv_val = max(0.5, float(srv) if srv not in [None,""] else 1)
            except: srv_val = 1
            entries.append((meal_name, str(food).strip(), srv_val))

    if not entries:
        return None

    totals = {k:0 for k in ["calories","protein","carbs","fats","fiber","sugar","sodium_mg"]}
    processed_count = 0
    meal_lines, match_notes, diet_flags = [], [], []
    food_log = []

    for meal_name, food_name, servings in entries:
        row = get_food(food_name, api_key)
        n = compute_nutrients(row, servings)
        for k in totals:
            totals[k] += n[k]

        if row.get("category") == "processed":
            processed_count += 1

        # Diet compatibility
        tags = str(row.get("diet_tags","")).lower()
        ud = diet_pref.lower().replace("-","")
        if "vegan" in ud and "vegan" not in tags:
            diet_flags.append(f"{row['food_name']} (not vegan)")
        elif "vegetarian" in ud and "non-veg" in tags:
            diet_flags.append(f"{row['food_name']} (non-vegetarian)")

        label = row.get("serving_label","1 serving")
        srv_str = f"{servings:.1f}".rstrip('0').rstrip('.')
        meal_lines.append(f"{meal_name}: {row['food_name']} × {srv_str} [{label}] = {round(n['calories'])} kcal")
        food_log.append({"food":row["food_name"],"meal":meal_name,"servings":servings,**n})

        mt = row.get("match_type","")
        if mt != "exact":
            match_notes.append(f"'{food_name}' → {mt}")

    totals = {k: round(v, 1) for k, v in totals.items()}

    # ── Issue detection (8 rules, all grounded in nutrition science) ─────
    issues = []
    # Rule 1 — Protein (WHO: 0.8g/kg; 30g/day minimum for avg adult)
    if totals["protein"] < 30:
        issues.append(("Low protein", "Insufficient protein (<30g) may impair muscle maintenance and satiety."))
    # Rule 2 — Fiber (WHO recommendation: 25-38g/day)
    if totals["fiber"] < 10:
        issues.append(("Low fiber", "Low fiber (<10g) is linked to poor digestive health and increased disease risk."))
    # Rule 3 — Sugar (WHO: <50g free sugar/day)
    if totals["sugar"] > 50:
        issues.append(("High sugar", "Excess sugar (>50g) raises risk of obesity, diabetes, and tooth decay."))
    # Rule 4 — Sodium (WHO: <2000mg/day)
    if totals["sodium_mg"] > 2000:
        issues.append(("High sodium", "High sodium (>2000mg) increases blood pressure and cardiovascular risk."))
    # Rule 5 — Processed foods
    if processed_count >= 1:
        issues.append(("Processed food present", "Ultra-processed foods are associated with poor long-term health outcomes."))
    # Rule 6 — Carbohydrate balance
    if totals["protein"] > 0 and totals["carbs"] > totals["protein"] * 3.5:
        issues.append(("Carb-heavy pattern", "High carb-to-protein ratio may reduce satiety and metabolic efficiency."))
    # Rule 7 — Total calories
    if totals["calories"] > 2500:
        issues.append(("High calorie intake", "Total calories exceed typical daily requirements for most adults."))
    if totals["calories"] < 800:
        issues.append(("Very low calorie intake", "Calorie intake is very low, which may cause nutrient deficiencies."))
    # Rule 8 — Diet compatibility
    if diet_flags:
        issues.append(("Diet preference mismatch", "Some foods may not align with your stated diet preference."))

    status = "Well Balanced" if not issues else ("Needs Improvement" if len(issues) <= 2 else "Poorly Balanced")

    return {
        "diet_preference": diet_pref,
        "goal":            goal,
        "meal_lines":      meal_lines,
        "food_log":        food_log,
        "match_notes":     match_notes,
        "diet_flags":      diet_flags,
        "totals":          totals,
        "processed_count": processed_count,
        "issues":          issues,
        "status":          status,
    }

# ─── SESSION TRACKER — longitudinal seven-analysis log ─────────────────────────────
class SessionTracker:
    """
    Maintains multi-turn conversation history and a seven-analysis nutritional log.
    This implements the memory layer of the agent — enabling longitudinal
    behavioral pattern analysis across multiple interactions.
    """
    def __init__(self):
        self.conversation_history: list[dict] = []
        self.daily_logs: list[dict] = []        # list of daily facts dicts
        self.session_count: int = 0
        self.nudge_log: list[str] = []          # tracks issued nudges

    def add_day(self, facts: dict):
        entry = {
            "date": datetime.date.today().isoformat(),
            "session": self.session_count,
            **facts["totals"],
            "status": facts["status"],
            "issues": [i[0] for i in facts["issues"]],
            "foods": [f["food"] for f in facts["food_log"]],
        }
        self.daily_logs.append(entry)
        # Keep last 7 analyses only
        if len(self.daily_logs) > 7:
            self.daily_logs = self.daily_logs[-7:]
        self.session_count += 1

    def get_weekly_trends(self) -> dict:
        if not self.daily_logs:
            return {}
        n = len(self.daily_logs)
        keys = ["calories","protein","carbs","fats","fiber","sugar","sodium_mg"]
        avgs = {k: round(sum(d.get(k,0) for d in self.daily_logs)/n, 1) for k in keys}
        all_issues = [i for d in self.daily_logs for i in d.get("issues",[])]
        from collections import Counter
        top_issues = [i for i,_ in Counter(all_issues).most_common(3)]
        return {"analyses_tracked": n, "averages": avgs, "recurring_issues": top_issues}

    def reset(self):
        self.__init__()



# ─── SYSTEM PROMPT — the cognitive architecture ────────────────────────────
# Each instruction maps directly to a theoretical framework:
# CLT = Cognitive Load Theory | HBM = Health Belief Model
# NBT = Nudge Theory          | IFT = Information Foraging Theory

SYSTEM_PROMPT = """You are a Cognitively Aware AI Nutrition Decision Assistant, grounded in four behavioral theories.

THEORETICAL PRINCIPLES — apply all four in every response:

1. Cognitive Load Theory (Sweller, 1988) [CLT]
   - Maximum 3 recommendations per response — never more
   - Use plain, jargon-free language
   - Present information in structured sections to reduce extraneous load
   - Total response must stay under 300 words

2. Health Belief Model (Rosenstock, 1974) [HBM]
   - For each imbalance, state the specific health risk (perceived susceptibility)
   - Frame severity clearly but without alarmism
   - Connect recommendations to the user's personal goal (perceived benefit)
   - Never just name an issue — always explain why it matters for THIS user

3. Behavioral Nudge Theory (Thaler & Sunstein, 2008) [NBT]
   - End with exactly ONE nudge — a single micro-habit for tomorrow
   - The nudge must feel easy, specific, and immediately actionable
   - If longitudinal data is available, reference prior patterns in the nudge
   - Never issue more than one nudge per session

4. Information Foraging Theory (Pirolli & Card, 1999) [IFT]
   - Filter output to the user's stated goal — suppress irrelevant information
   - For Weight Loss: emphasize protein, fiber, calories
   - For Muscle Gain: emphasize protein, carb timing, total intake
   - For Blood Sugar Control: emphasize glycemic index, sugar, fiber
   - For Heart Health: emphasize sodium, fats, fiber
   - For Energy Improvement: emphasize carb quality, iron-rich foods, meal timing

REQUIRED OUTPUT FORMAT — use this structure exactly:

### 🍽 Nutrition Summary
[2-3 warm sentences. Name what they ate, show actual calorie total, overall status. Non-judgmental.]

### ⚠️ Issues Detected
[1-3 bullets. Format: **Issue name** — health risk in one sentence. Goal-relevant only.]

### ✅ Personalized Recommendations
[1-3 bullets. Specific foods. Respect diet preference. Connected to goal.]

### 💡 Nudge for Tomorrow
[ONE sentence only. Easy, specific, achievable.]

RULES:
- Never give medical diagnoses or replace professional advice
- If meals are already well balanced, say so warmly and give a positive reinforcement nudge
- In follow-up questions, maintain conversation context naturally
- Warm, encouraging, non-shaming tone always
"""

# ─── LLM CALL ─────────────────────────────────────────────────────────────
def call_llm_analysis(facts: dict, api_key: str, weekly_trends: dict, session: SessionTracker) -> str:
    t = facts["totals"]
    issues_str = "\n".join(f"  - {i[0]}: {i[1]}" for i in facts["issues"]) or "  - None detected"
    trends_str = ""
    if weekly_trends.get("analyses_tracked", 0) > 1:
        avgs = weekly_trends["averages"]
        rec = weekly_trends.get("recurring_issues", [])
        trends_str = f"""
LONGITUDINAL CONTEXT ({weekly_trends['analyses_tracked']}-day history):
  - Avg daily calories: {avgs['calories']} kcal
  - Avg protein: {avgs['protein']}g | Avg fiber: {avgs['fiber']}g | Avg sugar: {avgs['sugar']}g
  - Recurring issues across sessions: {', '.join(rec) if rec else 'None'}
  → Use this to personalize the nudge and recommendations."""

    user_msg = f"""Analyze this user's nutrition and provide cognitively structured guidance.

USER PROFILE:
  - Diet preference: {facts['diet_preference']}
  - Health goal: {facts['goal']}

TODAY'S MEALS (with actual serving sizes and kcal):
{chr(10).join('  • ' + line for line in facts['meal_lines'])}

DAILY NUTRITIONAL TOTALS:
  - Calories:  {t['calories']} kcal
  - Protein:   {t['protein']} g
  - Carbs:     {t['carbs']} g
  - Fats:      {t['fats']} g
  - Fiber:     {t['fiber']} g
  - Sugar:     {t['sugar']} g
  - Sodium:    {t['sodium_mg']} mg

RULE ENGINE FLAGS:
{issues_str}
Overall status: {facts['status']}
{trends_str}

Apply all four theoretical frameworks. Tailor to goal: {facts['goal']}. Respect diet: {facts['diet_preference']}."""

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=session.conversation_history + [{"role":"user","content":user_msg}],
    )
    return resp.content[0].text

def call_llm_followup(user_question: str, api_key: str, session: SessionTracker) -> str:
    """Handle conversational follow-up questions using session memory."""
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=session.conversation_history + [{"role":"user","content":user_question}],
    )
    return resp.content[0].text

def parse_llm_sections(text: str) -> tuple[str,str,str,str]:
    """Extract four structured sections from LLM response."""
    def extract(t, heading):
        marker = f"### {heading}"
        if marker not in t: return ""
        start = t.index(marker)
        rest = t[start+len(marker):].strip()
        nxt = rest.find("\n### ")
        return marker + "\n" + (rest[:nxt].strip() if nxt!=-1 else rest.strip())
    summary = extract(text,"🍽 Nutrition Summary") or text
    issues  = extract(text,"⚠️ Issues Detected")
    recs    = extract(text,"✅ Personalized Recommendations")
    nudge   = extract(text,"💡 Nudge for Tomorrow")
    return summary, issues, recs, nudge

# ─── WEEKLY SUMMARY GENERATOR ─────────────────────────────────────────────
def generate_weekly_summary(api_key: str, session: SessionTracker) -> str:
    if not session.daily_logs:
        return "No data logged yet. Analyze at least one day first."
    trends = session.get_weekly_trends()
    avgs = trends["averages"]
    prompt = f"""Generate a concise weekly nutrition summary for a user.

DATA ({trends['analyses_tracked']} days tracked):
Average daily: {avgs['calories']} kcal | Protein: {avgs['protein']}g | Carbs: {avgs['carbs']}g | Fats: {avgs['fats']}g | Fiber: {avgs['fiber']}g | Sugar: {avgs['sugar']}g | Sodium: {avgs['sodium_mg']}mg
Recurring issues: {', '.join(trends.get('recurring_issues',[])) or 'None'}

Write a 150-word warm, encouraging weekly summary with:
1. What they did well this week
2. The one most important area to improve next week
3. One specific weekly goal (not daily — something to achieve over 7 days)"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL, max_tokens=400,
            messages=[{"role":"user","content":prompt}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"Error generating summary: {e}"

