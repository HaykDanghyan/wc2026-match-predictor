import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
from collections import defaultdict
from scipy.stats import poisson
from xgboost import XGBRegressor
import pickle, os, warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* dark pitch background */
.stApp { background: #0a1628; color: #e8edf5; }

/* hero banner */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 50%, #0d2137 100%);
    border: 1px solid #1e4976;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "⚽";
    position: absolute;
    font-size: 12rem;
    right: -1rem;
    top: -2rem;
    opacity: 0.04;
}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 3px;
    color: #f5c842;
    margin: 0;
    line-height: 1;
}
.hero p { color: #8aa8c8; font-size: 1rem; margin: 0.5rem 0 0 0; }

/* section headers */
.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 2px;
    color: #f5c842;
    border-bottom: 2px solid #1e4976;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

/* metric cards */
.metric-card {
    background: #0d2137;
    border: 1px solid #1e4976;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .label { color: #8aa8c8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-family: 'Bebas Neue', sans-serif; font-size: 2.6rem; color: #f5c842; line-height: 1.1; }
.metric-card .sub   { color: #c8d8e8; font-size: 0.85rem; margin-top: 0.2rem; }

/* team selector */
.vs-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #f5c842;
    text-align: center;
    padding-top: 1.8rem;
}

/* scoreline display */
.scoreline {
    background: linear-gradient(135deg, #0d2137, #1a3a5c);
    border: 2px solid #f5c842;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.scoreline .score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: #f5c842;
    letter-spacing: 6px;
    line-height: 1;
}
.scoreline .label { color: #8aa8c8; font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase; }

/* probability bar */
.prob-bar-container { margin: 0.4rem 0; }
.prob-bar-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #8aa8c8; margin-bottom: 2px; }

/* group table */
.group-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 2px;
    color: #f5c842;
    background: #0d2137;
    border: 1px solid #1e4976;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
}

/* qualification badge */
.badge-qualify { background: #1a4a2e; color: #4ade80; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600; }
.badge-maybe   { background: #3a3010; color: #fbbf24; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600; }
.badge-out     { background: #3a1010; color: #f87171; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600; }

/* tab styling */
button[data-baseweb="tab"] { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 1.5px !important; font-size: 1rem !important; }

/* dataframe overrides */
.stDataFrame { border: 1px solid #1e4976 !important; border-radius: 8px !important; }

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
ALL_WC_TEAMS = sorted([t for teams in GROUPS.values() for t in teams])

FEATURE_COLS = [
    'elo_diff','home_elo','away_elo',
    'home_form_gf','home_form_ga',
    'away_form_gf','away_form_ga',
    'neutral','match_importance'
]

# ── MODEL + STATE LOADING ─────────────────────────────────────────────────────
@st.cache_resource
def load_models_and_state():
    """
    Load pre-trained models and current team state.
    If pickle files don't exist yet, show instructions.
    """
    try:
        with open("wc2026_models.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle["home_model"], bundle["away_model"], bundle["current_state"], True
    except FileNotFoundError:
        return None, None, None, False

home_model, away_model, current_state, models_loaded = load_models_and_state()

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def build_feature_row(team_a, team_b, neutral=True, match_importance=3):
    a = current_state[team_a]
    b = current_state[team_b]
    row = {
        "elo_diff":         a["elo"] - b["elo"],
        "home_elo":         a["elo"],
        "away_elo":         b["elo"],
        "home_form_gf":     a["form_gf"],
        "home_form_ga":     a["form_ga"],
        "away_form_gf":     b["form_gf"],
        "away_form_ga":     b["form_ga"],
        "neutral":          neutral,
        "match_importance": match_importance,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

def predict_score_probabilities(home_xg, away_xg, max_goals=8):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i,j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    probs /= probs.sum()
    p_home = np.tril(probs,-1).sum()
    p_draw  = np.trace(probs)
    p_away  = np.triu(probs, 1).sum()
    best    = np.unravel_index(np.argmax(probs), probs.shape)
    return p_home, p_draw, p_away, best, probs

def predict_match(team_a, team_b, neutral=True, match_importance=3):
    X = build_feature_row(team_a, team_b, neutral, match_importance)
    lam_a = float(home_model.predict(X)[0])
    lam_b = float(away_model.predict(X)[0])
    p_a, p_d, p_b, best, matrix = predict_score_probabilities(lam_a, lam_b)
    return {
        "team_a": team_a, "team_b": team_b,
        "lam_a": lam_a, "lam_b": lam_b,
        "total_goals": lam_a + lam_b,
        "score": best,
        "p_a": p_a, "p_draw": p_d, "p_b": p_b,
        "matrix": matrix,
    }

@st.cache_data(show_spinner=False)
def precompute_lambdas():
    lambdas = {}
    for group, teams in GROUPS.items():
        for ta, tb in itertools.combinations(teams, 2):
            X = build_feature_row(ta, tb, neutral=True, match_importance=3)
            lambdas[(ta,tb)] = (float(home_model.predict(X)[0]),
                                float(away_model.predict(X)[0]))
    return lambdas

@st.cache_data(show_spinner=False)
def run_monte_carlo(n_sims=5000, seed=42):
    lambdas = precompute_lambdas()
    rng = np.random.default_rng(seed)

    winner_c   = defaultdict(int)
    runnerup_c = defaultdict(int)
    third_c    = defaultdict(int)
    qualify_c  = defaultdict(int)
    pts_acc    = defaultdict(list)

    for _ in range(n_sims):
        third_candidates = []
        for group, teams in GROUPS.items():
            pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int)
            for ta, tb in itertools.combinations(teams, 2):
                key = (ta,tb) if (ta,tb) in lambdas else (tb,ta)
                la, lb = lambdas[key]
                ga_ = rng.poisson(la); gb_ = rng.poisson(lb)
                gf[ta]+=ga_; ga[ta]+=gb_; gf[tb]+=gb_; ga[tb]+=ga_
                if ga_>gb_: pts[ta]+=3
                elif gb_>ga_: pts[tb]+=3
                else: pts[ta]+=1; pts[tb]+=1
            gd = {t: gf[t]-ga[t] for t in teams}
            ranking = sorted(teams, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
            winner_c[ranking[0]]+=1; runnerup_c[ranking[1]]+=1
            qualify_c[ranking[0]]+=1; qualify_c[ranking[1]]+=1
            for t in teams: pts_acc[t].append(pts[t])
            third = ranking[2]
            third_candidates.append((third, pts[third], gd[third], gf[third]))
        third_candidates.sort(key=lambda x:(x[1],x[2],x[3]), reverse=True)
        for team,*_ in third_candidates[:8]:
            third_c[team]+=1; qualify_c[team]+=1

    n = n_sims
    rows = []
    for group, teams in GROUPS.items():
        for t in teams:
            rows.append({
                "Group": group, "Team": t,
                "Elo": round(current_state[t]["elo"],0),
                "P(1st)": round(winner_c[t]/n*100,1),
                "P(2nd)": round(runnerup_c[t]/n*100,1),
                "P(Best 3rd)": round(third_c[t]/n*100,1),
                "P(Qualify)": round(qualify_c[t]/n*100,1),
                "Avg Pts": round(np.mean(pts_acc[t]),2),
            })
    return pd.DataFrame(rows)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏆 World Cup 2026 Predictor</h1>
    <p>XGBoost Poisson model · Elo ratings · Monte Carlo group simulation · 48 teams · 12 groups</p>
</div>
""", unsafe_allow_html=True)

# ── SETUP GUARD ───────────────────────────────────────────────────────────────
if not models_loaded:
    st.error("⚠️  Model file not found. Run the cell below in your notebook first, then restart the app.")
    st.code("""
import pickle
bundle = {
    "home_model":    home_goal_model,
    "away_model":    away_goal_model,
    "current_state": current_state,
}
with open("wc2026_models.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("Saved!")
    """, language="python")
    st.stop()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚽  Match Predictor", "🗂️  Group Stage", "🏅  Qualification Ranking"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">PREDICT ANY MATCH</div>', unsafe_allow_html=True)

    col1, colv, col2 = st.columns([5,1,5])
    with col1:
        team_a = st.selectbox("Home / Team A", ALL_WC_TEAMS,
                              index=ALL_WC_TEAMS.index("Argentina"), key="ta")
    with colv:
        st.markdown('<div class="vs-label">VS</div>', unsafe_allow_html=True)
    with col2:
        team_b = st.selectbox("Away / Team B", ALL_WC_TEAMS,
                              index=ALL_WC_TEAMS.index("France"), key="tb")

    c1, c2 = st.columns(2)
    with c1:
        neutral = st.toggle("Neutral venue (World Cup default)", value=True)
    with c2:
        importance_map = {"World Cup (3)":3, "Qualifier (2)":2, "Friendly (0)":0, "Other (1)":1}
        imp_label = st.selectbox("Match importance", list(importance_map.keys()))
        match_importance = importance_map[imp_label]

    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        if st.button("🔮  Predict Match", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                r = predict_match(team_a, team_b, neutral, match_importance)

            # scoreline
            st.markdown(f"""
            <div class="scoreline">
                <div class="label">MOST LIKELY SCORELINE</div>
                <div class="score">{r['score'][0]} — {r['score'][1]}</div>
                <div class="label">{team_a} &nbsp;·&nbsp; {team_b}</div>
            </div>""", unsafe_allow_html=True)

            # metrics row
            m1,m2,m3,m4 = st.columns(4)
            m1.markdown(f"""<div class="metric-card">
                <div class="label">Expected Goals</div>
                <div class="value">{r['lam_a']:.2f}</div>
                <div class="sub">{team_a}</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card">
                <div class="label">Expected Goals</div>
                <div class="value">{r['lam_b']:.2f}</div>
                <div class="sub">{team_b}</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card">
                <div class="label">Total Goals</div>
                <div class="value">{r['total_goals']:.2f}</div>
                <div class="sub">predicted</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card">
                <div class="label">Elo Diff</div>
                <div class="value">{current_state[team_a]['elo']-current_state[team_b]['elo']:+.0f}</div>
                <div class="sub">A minus B</div></div>""", unsafe_allow_html=True)

            st.markdown("---")

            # probability bars
            st.markdown('<div class="section-label">OUTCOME PROBABILITIES</div>', unsafe_allow_html=True)
            pb1, pb2, pb3 = st.columns(3)
            pb1.metric(f"🟢 {team_a} Win", f"{r['p_a']*100:.1f}%")
            pb2.metric("🟡 Draw",           f"{r['p_draw']*100:.1f}%")
            pb3.metric(f"🔴 {team_b} Win",  f"{r['p_b']*100:.1f}%")

            # probability bar chart
            fig, ax = plt.subplots(figsize=(8,1.2))
            fig.patch.set_facecolor('#0a1628')
            ax.set_facecolor('#0a1628')
            bars = [r['p_a'], r['p_draw'], r['p_b']]
            colors = ['#4ade80','#fbbf24','#f87171']
            left = 0
            for val, col in zip(bars, colors):
                ax.barh(0, val, left=left, color=col, height=0.5)
                if val > 0.07:
                    ax.text(left + val/2, 0, f"{val*100:.1f}%",
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='#0a1628')
                left += val
            ax.set_xlim(0,1); ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("---")

            # scoreline probability matrix
            st.markdown('<div class="section-label">SCORELINE PROBABILITY MATRIX (%)</div>', unsafe_allow_html=True)
            st.caption(f"Rows = {team_a} goals · Columns = {team_b} goals")
            matrix_pct = np.round(r['matrix'] * 100, 1)
            matrix_df = pd.DataFrame(
                matrix_pct,
                index=[f"{team_a} {i}" for i in range(matrix_pct.shape[0])],
                columns=[f"{team_b} {j}" for j in range(matrix_pct.shape[1])]
            )
            st.dataframe(matrix_df.style.background_gradient(cmap="YlOrRd", axis=None),
                         use_container_width=True)

            # team current state
            st.markdown("---")
            st.markdown('<div class="section-label">TEAM STATS GOING INTO THIS MATCH</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            with s1:
                sa = current_state[team_a]
                st.markdown(f"**{team_a}**")
                st.write(f"Elo rating: **{sa['elo']:.0f}**")
                st.write(f"Avg goals scored (last 5): **{sa['form_gf']:.1f}**")
                st.write(f"Avg goals conceded (last 5): **{sa['form_ga']:.1f}**")
            with s2:
                sb = current_state[team_b]
                st.markdown(f"**{team_b}**")
                st.write(f"Elo rating: **{sb['elo']:.0f}**")
                st.write(f"Avg goals scored (last 5): **{sb['form_gf']:.1f}**")
                st.write(f"Avg goals conceded (last 5): **{sb['form_ga']:.1f}**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GROUP STAGE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">GROUP STAGE PREDICTIONS</div>', unsafe_allow_html=True)
    st.caption("Based on 5,000 Monte Carlo simulations using your trained XGBoost Poisson models.")

    n_sims = st.slider("Simulation runs", 1000, 10000, 5000, step=1000)

    if st.button("🎲  Run Group Stage Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating group stage..."):
            qual_df = run_monte_carlo.clear() or run_monte_carlo.__wrapped__(n_sims=n_sims)

        # display by group in 2-column grid
        groups = sorted(GROUPS.keys())
        for i in range(0, len(groups), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i+j >= len(groups): break
                g = groups[i+j]
                gdf = qual_df[qual_df["Group"]==g].sort_values("P(Qualify)", ascending=False)
                with col:
                    st.markdown(f'<div class="group-header">GROUP {g}</div>', unsafe_allow_html=True)
                    for _, row in gdf.iterrows():
                        pq = row["P(Qualify)"]
                        if pq >= 70:
                            badge = f'<span class="badge-qualify">▲ {pq}%</span>'
                        elif pq >= 35:
                            badge = f'<span class="badge-maybe">~ {pq}%</span>'
                        else:
                            badge = f'<span class="badge-out">▼ {pq}%</span>'
                        st.markdown(
                            f"**{row['Team']}** &nbsp; {badge} &nbsp; "
                            f"<span style='color:#8aa8c8;font-size:0.8rem'>"
                            f"1st {row['P(1st)']}% · 2nd {row['P(2nd)']}% · 3rd {row['P(Best 3rd)']}%"
                            f"</span>",
                            unsafe_allow_html=True
                        )
                    st.markdown("<br>", unsafe_allow_html=True)

        # group-by-group predicted matches
        st.markdown("---")
        st.markdown('<div class="section-label">ALL GROUP STAGE MATCH PREDICTIONS</div>', unsafe_allow_html=True)
        selected_group = st.selectbox("Select group to view matches", sorted(GROUPS.keys()))
        teams = GROUPS[selected_group]
        match_rows = []
        for ta, tb in itertools.combinations(teams, 2):
            r = predict_match(ta, tb, neutral=True, match_importance=3)
            match_rows.append({
                "Match": f"{ta} vs {tb}",
                "Score": f"{r['score'][0]}-{r['score'][1]}",
                "xG A": f"{r['lam_a']:.2f}",
                "xG B": f"{r['lam_b']:.2f}",
                "Total xG": f"{r['total_goals']:.2f}",
                f"{ta} Win %": f"{r['p_a']*100:.1f}%",
                "Draw %": f"{r['p_draw']*100:.1f}%",
                f"{tb} Win %": f"{r['p_b']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(match_rows), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FULL QUALIFICATION RANKING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">FULL TOURNAMENT QUALIFICATION RANKING</div>', unsafe_allow_html=True)
    st.caption("All 48 teams ranked by probability of advancing to the Round of 32.")

    if st.button("🏅  Generate Full Ranking", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            qual_df = run_monte_carlo(n_sims=5000)

        # top 32 qualifiers
        top32 = qual_df.sort_values("P(Qualify)", ascending=False).reset_index(drop=True)
        top32.index += 1

        def color_qualify(val):
            if val >= 70: return "color: #4ade80; font-weight: 600"
            elif val >= 35: return "color: #fbbf24"
            else: return "color: #f87171"

        st.dataframe(
            top32[["Group","Team","Elo","P(1st)","P(2nd)","P(Best 3rd)","P(Qualify)","Avg Pts"]]
            .style.map(color_qualify, subset=["P(Qualify)"]),
            use_container_width=True, height=600
        )

        # bar chart top 20
        st.markdown("---")
        st.markdown('<div class="section-label">TOP 20 TEAMS BY QUALIFICATION PROBABILITY</div>', unsafe_allow_html=True)
        top20 = top32.head(20)
        fig, ax = plt.subplots(figsize=(10,6))
        fig.patch.set_facecolor('#0d2137')
        ax.set_facecolor('#0d2137')
        bars = ax.barh(top20["Team"][::-1], top20["P(Qualify)"][::-1],
                       color='#f5c842', edgecolor='none', height=0.6)
        for bar, val in zip(bars, top20["P(Qualify)"][::-1]):
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f"{val}%", va='center', color='#e8edf5', fontsize=9)
        ax.set_xlabel("P(Qualify) %", color='#8aa8c8')
        ax.tick_params(colors='#e8edf5', labelsize=9)
        ax.spines[['top','right','bottom']].set_visible(False)
        ax.spines['left'].set_color('#1e4976')
        ax.set_xlim(0,110)
        ax.axvline(50, color='#1e4976', linestyle='--', linewidth=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()