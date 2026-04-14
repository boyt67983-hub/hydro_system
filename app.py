import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ============================================================
# Modern Hydro Data System UI Upgrade
# Save as: app.py
# Run: streamlit run app.py
# Requires: model.pkl in the same folder
# ============================================================

st.set_page_config(
    page_title="Hydro Data Intelligence",
    page_icon="images.png",

    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Modern theme / CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.20), transparent 25%),
        radial-gradient(circle at top right, rgba(59,130,246,0.18), transparent 22%),
        linear-gradient(135deg, #07111f 0%, #0b1730 40%, #0f2745 100%);
    color: #e5eefc;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,15,28,0.96), rgba(14,29,56,0.96));
    border-right: 1px solid rgba(148, 163, 184, 0.16);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

.hero {
    background: linear-gradient(135deg, rgba(14,165,233,0.16), rgba(59,130,246,0.16));
    border: 1px solid rgba(125, 211, 252, 0.18);
    backdrop-filter: blur(14px);
    border-radius: 26px;
    padding: 28px 28px 22px 28px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.24);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    color: #f8fbff;
}

.hero-sub {
    font-size: 1rem;
    color: #bfd4f7;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #dbeafe;
    margin: 1.2rem 0 0.75rem 0;
    padding-left: 0.85rem;
    border-left: 4px solid #38bdf8;
}

.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 22px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    backdrop-filter: blur(12px);
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #93c5fd;
    margin-bottom: 0.35rem;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
}

.metric-unit {
    font-size: 0.82rem;
    color: #cbd5e1;
}

.panel {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.16);
    margin-bottom: 1rem;
}

.kpi-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 0.5rem;
}

.kpi-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 16px;
    padding: 12px 14px;
}

.kpi-title {
    font-size: 0.78rem;
    color: #93c5fd;
}

.kpi-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
}

.status-good, .status-high, .status-medium {
    border-radius: 18px;
    padding: 16px 18px;
    margin: 0.8rem 0;
    border: 1px solid transparent;
}

.status-good {
    background: rgba(34,197,94,0.12);
    color: #dcfce7;
    border-color: rgba(34,197,94,0.24);
}

.status-high {
    background: rgba(239,68,68,0.12);
    color: #fee2e2;
    border-color: rgba(239,68,68,0.24);
}

.status-medium {
    background: rgba(245,158,11,0.12);
    color: #fef3c7;
    border-color: rgba(245,158,11,0.24);
}

.small-note {
    color: #93a8c9;
    font-size: 0.88rem;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.14);
}

.stButton > button {
    border-radius: 14px;
    font-weight: 700;
    border: none;
    padding: 0.7rem 1rem;
    box-shadow: 0 10px 24px rgba(14,165,233,0.22);
}

.stSelectbox label, .stNumberInput label, .stRadio label {
    color: #dbeafe !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load model bundle
# -----------------------------
@st.cache_resource(show_spinner="Loading model bundle...")
def load_bundle():
    return joblib.load("model.pkl")

try:
    B = load_bundle()
except FileNotFoundError:
    st.error("model.pkl not found. Place it beside app.py and rerun the app.")
    st.stop()

model           = B['model']
FEATURES        = B['features']
TARGETS         = B['targets']
MONTH_MAP       = B['month_map']
thresholds      = B['thresholds']
monthly_stats   = B['monthly_stats']
monthly_rolling = B['monthly_rolling']
recent_data     = pd.DataFrame(B['recent_rows'])
season_map      = B['season_map']

MONTHS_LIST = list(MONTH_MAP.keys())
MONTH_NAMES = {v: k for k, v in MONTH_MAP.items()}
MAX_DAYS    = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}


# -----------------------------
# Core helpers
# -----------------------------
def get_context(month_num, date):
    before = recent_data[
        (recent_data['Month_num'] < month_num) |
        ((recent_data['Month_num'] == month_num) & (recent_data['Date'] < date))
    ].tail(3)
    return before


def build_feature_row(month_num, date):
    ctx = get_context(month_num, date)

    if ctx.empty:
        pm  = monthly_stats['Morning'].get(month_num, 40)
        pn  = monthly_stats['Noon'].get(month_num, 45)
        pa  = monthly_stats['Afternoon'].get(month_num, 45)
        pd_ = monthly_stats['Discharge'].get(month_num, 50)
    else:
        last = ctx.iloc[-1]
        pm, pn, pa, pd_ = last['Morning'], last['Noon'], last['Afternoon'], last['Discharge']

    smean = lambda col: ctx[col].mean() if len(ctx) > 0 else monthly_stats[col].get(month_num, 40)
    sstd  = lambda col: ctx[col].std() if len(ctx) > 1 else 5.0

    feat = {
        'Month_num': month_num,
        'Date': date,
        'Season': season_map.get(month_num, 1),
        'prev_Morning': pm,
        'prev_Noon': pn,
        'prev_Afternoon': pa,
        'prev_Discharge': pd_,
        'roll_mean_Morning': smean('Morning'),
        'roll_mean_Noon': smean('Noon'),
        'roll_mean_Afternoon': smean('Afternoon'),
        'roll_mean_Discharge': smean('Discharge'),
        'roll_std_Morning': sstd('Morning'),
        'roll_std_Discharge': sstd('Discharge'),
    }
    return np.array([[feat[f] for f in FEATURES]])


def predict(month_num, date):
    p = model.predict(build_feature_row(month_num, date))[0]
    return {
        'Morning': round(max(p[0], 1), 1),
        'Noon': round(max(p[1], 1), 1),
        'Afternoon': round(max(p[2], 1), 1),
        'Average': round(max(p[3], 0.001), 4),
        'Discharge': round(max(p[4], 0.001), 4),
    }


def check_anomalies(month_num, date, vals):
    H, M = [], []
    m, n, a, avg, dis = vals['Morning'], vals['Noon'], vals['Afternoon'], vals['Average'], vals['Discharge']
    ctx = get_context(month_num, date)

    if len(ctx) >= 2:
        for col, v in [('Morning', m), ('Noon', n), ('Afternoon', a), ('Discharge', dis)]:
            mean = ctx[col].mean()
            std = ctx[col].std()
            if std > 0:
                z = abs((v - mean) / std)
                direction = 'higher' if v > mean else 'lower'
                if z > 5:
                    H.append(f"{col}: {v} is far {direction} than recent days (avg={mean:.1f}, z={z:.1f})")
                elif z > 2.5:
                    M.append(f"{col}: {v} is slightly {direction} than recent days (avg={mean:.1f}, z={z:.1f})")

    mr = monthly_rolling.get(month_num, {})
    for col, v in [('Morning', m), ('Noon', n), ('Afternoon', a), ('Discharge', dis)]:
        s = mr.get(col, {})
        if s and s['std'] > 0:
            z = abs((v - s['mean']) / s['std'])
            if z > 4:
                H.append(f"{col}: {v} is very unusual for {MONTH_NAMES.get(month_num, '')} (avg={s['mean']:.1f}, z={z:.1f})")

    expected = round((m + n + a) / 3 / 100, 4)
    if abs(avg - expected) > 0.005:
        H.append(f"Average mismatch: entered {avg:.4f}, expected {expected:.4f}")

    if len(ctx) >= 1:
        prev_avg = ctx.iloc[-1]['Average']
        prev_dis = ctx.iloc[-1]['Discharge']
        if avg > prev_avg * 1.15 and dis < prev_dis * 0.85:
            H.append("Discharge dropped while gauge height rose — physically inconsistent")
        if avg < prev_avg * 0.85 and dis > prev_dis * 1.15:
            H.append("Discharge rose while gauge height dropped — physically inconsistent")

    for col, v in [('Morning', m), ('Noon', n), ('Afternoon', a), ('Discharge', dis)]:
        t = thresholds.get(col, {})
        if t and v > t['max'] * 3:
            H.append(f"{col}: {v} exceeds 3× historical max ({t['max']:.0f})")

    return H, M


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("images.png", width=120)
    st.markdown("## 💧 Hydro Intelligence")
    st.caption("Smart prediction + anomaly detection")
    st.markdown("---")

    mode = st.radio(
        "Choose workspace",
        ["🔮 Predict Missing Day", "🧪 Validate Entered Values"],
    )

    st.markdown("---")
    st.markdown("### Quick Facts")
    st.markdown("- 364 records across 12 months")
    st.markdown("- Seasonal + rolling features")
    st.markdown("- Random Forest model")

    chart_df = pd.DataFrame({
        'Month': MONTHS_LIST,
        'Avg Morning': [round(monthly_stats['Morning'].get(i, 0), 1) for i in range(1, 13)],
        'Avg Discharge': [round(monthly_stats['Discharge'].get(i, 0), 1) for i in range(1, 13)],
    }).set_index('Month')

    st.markdown("---")
    st.markdown("### Monthly Snapshot")
    st.bar_chart(chart_df)


# -----------------------------
# Header / hero
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">Hydro Data Intelligence Dashboard</div>
    <div class="hero-sub">Predict missing hydrological values, validate entered readings, and detect anomalies with a cleaner modern interface.</div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
k1.markdown(f'<div class="kpi-pill"><div class="kpi-title">Months Covered</div><div class="kpi-value">12</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-pill"><div class="kpi-title">Target Variables</div><div class="kpi-value">5 Outputs</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-pill"><div class="kpi-title">Prediction Engine</div><div class="kpi-value">Random Forest</div></div>', unsafe_allow_html=True)


# -----------------------------
# MODE 1
# -----------------------------
if "Predict" in mode:
    st.markdown('<div class="section-title">Predict values for a missing date</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Select month and day. The model uses recent context + seasonal patterns to estimate all target values.</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns([2, 1, 1.3])
    with p1:
        month = st.selectbox("Month", MONTHS_LIST)
        month_num = MONTH_MAP[month]
    with p2:
        date = st.number_input("Day", min_value=1, max_value=MAX_DAYS[month_num], value=1, step=1)
    with p3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Generate Prediction", type="primary", use_container_width=True)

    if predict_btn:
        result = predict(month_num, int(date))
        st.markdown(f'<div class="status-good">Prediction ready for <b>{month} {int(date)}</b>.</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        for col_ui, key, unit in [
            (c1, 'Morning', 'cm'),
            (c2, 'Noon', 'cm'),
            (c3, 'Afternoon', 'cm'),
            (c4, 'Average', 'm'),
            (c5, 'Discharge', 'm³/s'),
        ]:
            with col_ui:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value">{result[key]}</div>
                    <div class="metric-unit">{unit}</div>
                </div>
                """, unsafe_allow_html=True)

        ctx = get_context(month_num, int(date))
        left, right = st.columns([1.05, 1])

        with left:
            st.markdown('<div class="section-title">Recent rows used as context</div>', unsafe_allow_html=True)
            if not ctx.empty:
                ctx_display = ctx[['Month_num', 'Date'] + TARGETS].copy()
                ctx_display.insert(0, 'Month', ctx_display['Month_num'].map(MONTH_NAMES))
                ctx_display = ctx_display.drop(columns='Month_num')
                st.dataframe(ctx_display.reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                st.info("No previous rows found for this date, so monthly defaults were used.")

        with right:
            st.markdown('<div class="section-title">Monthly target trend</div>', unsafe_allow_html=True)
            trend = pd.DataFrame({
                'Month': MONTHS_LIST,
                'Avg Morning': [monthly_stats['Morning'].get(i, 0) for i in range(1, 13)],
                'Avg Noon': [monthly_stats['Noon'].get(i, 0) for i in range(1, 13)],
                'Avg Afternoon': [monthly_stats['Afternoon'].get(i, 0) for i in range(1, 13)],
            }).set_index('Month')
            st.line_chart(trend)


# -----------------------------
# MODE 2
# -----------------------------
else:
    st.markdown('<div class="section-title">Validate entered values and check anomalies</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Enter observed measurements and compare them against rule-based checks and model predictions.</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        month = st.selectbox("Month", MONTHS_LIST, key="chk_month")
        month_num = MONTH_MAP[month]
    with a2:
        date = st.number_input("Day", min_value=1, max_value=MAX_DAYS[month_num], value=1, step=1, key="chk_date")

    st.markdown('<div class="section-title">Gauge height inputs</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        morning = st.number_input("Morning (cm)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with g2:
        noon = st.number_input("Noon (cm)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with g3:
        afternoon = st.number_input("Afternoon (cm)", min_value=0.0, value=0.0, step=1.0, format="%.1f")

    auto_avg = round((morning + noon + afternoon) / 3 / 100, 4) if (morning + noon + afternoon) > 0 else 0.0
    x1, x2 = st.columns(2)
    with x1:
        avg_val = st.number_input(f"Average (m) · auto = {auto_avg}", min_value=0.0, value=auto_avg, step=0.0001, format="%.4f")
    with x2:
        discharge = st.number_input("Discharge", min_value=0.0, value=0.0, step=1.0, format="%.2f")

    check_btn = st.button("🔍 Run Quality Check", type="primary", use_container_width=True)

    if check_btn:
        if morning + noon + afternoon == 0:
            st.warning("Please enter gauge height values before running the check.")
        else:
            vals = {
                'Morning': morning,
                'Noon': noon,
                'Afternoon': afternoon,
                'Average': avg_val,
                'Discharge': discharge,
            }
            H, M = check_anomalies(month_num, int(date), vals)
            pred = predict(month_num, int(date))

            if not H and not M:
                st.markdown('<div class="status-good">✅ All checks passed. The data looks internally consistent.</div>', unsafe_allow_html=True)
            else:
                if H:
                    items = ''.join(f'<li>{item}</li>' for item in H)
                    st.markdown(f'<div class="status-high"><b>🔴 High severity issues</b><ul>{items}</ul></div>', unsafe_allow_html=True)
                if M:
                    items = ''.join(f'<li>{item}</li>' for item in M)
                    st.markdown(f'<div class="status-medium"><b>🟡 Medium severity issues</b><ul>{items}</ul></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Your values vs model prediction</div>', unsafe_allow_html=True)
            rows = []
            for t in TARGETS:
                entered = vals[t]
                modelled = pred[t]
                diff = entered - modelled
                pct = (diff / modelled * 100) if modelled != 0 else 0
                flag = '🔴' if abs(pct) > 30 else ('🟡' if abs(pct) > 15 else '🟢')
                rows.append({
                    'Field': t,
                    'You Entered': entered,
                    'Model Predicts': modelled,
                    'Difference': f'{diff:+.2f}',
                    '% Deviation': f'{pct:+.1f}%',
                    'Status': flag,
                })

            comp_df = pd.DataFrame(rows)

            def highlight(row):
                pct = float(row['% Deviation'].replace('%', '').replace('+', ''))
                if abs(pct) > 30:
                    return ['background-color: rgba(239,68,68,0.15)'] * len(row)
                if abs(pct) > 15:
                    return ['background-color: rgba(245,158,11,0.14)'] * len(row)
                return ['background-color: rgba(34,197,94,0.08)'] * len(row)

            st.dataframe(comp_df.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
            st.caption('🔴 >30% deviation · 🟡 >15% deviation · 🟢 acceptable range')

            if H:
                st.markdown('<div class="section-title">Suggested model-based replacement values</div>', unsafe_allow_html=True)
                s1, s2, s3, s4, s5 = st.columns(5)
                for col_ui, key, unit in [
                    (s1, 'Morning', 'cm'),
                    (s2, 'Noon', 'cm'),
                    (s3, 'Afternoon', 'cm'),
                    (s4, 'Average', 'm'),
                    (s5, 'Discharge', 'm³/s'),
                ]:
                    with col_ui:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{key}</div>
                            <div class="metric-value">{pred[key]}</div>
                            <div class="metric-unit">{unit}</div>
                        </div>
                        """, unsafe_allow_html=True)
