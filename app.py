import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import time
import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings("ignore")

# ─── Real Alert Sending Functions ──────────────────────────────────────────────
def send_real_email(to_email, subject, body, sender_email, sender_password):
    """Send actual email via Gmail SMTP (SSL port 465)."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender_email
        msg["To"]      = to_email

        html_body = f"""
        <html><body style="background:#0a0e1a;color:#c8d8e8;font-family:monospace;padding:20px">
        <h2 style="color:#00f5ff">⚙ PredictaMaint AI Alert</h2>
        <pre style="background:#0f1629;padding:15px;border-left:4px solid #ff6b00;
                    color:#c8d8e8;white-space:pre-wrap">{body}</pre>
        <p style="color:#7a9cc4;font-size:12px">— PredictaMaint AI v2.5.0 | Automated Alert System</p>
        </body></html>
        """
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True, "✅ Email delivered successfully."
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Gmail authentication failed. Check your email & App Password."
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error: {e}"
    except Exception as e:
        return False, f"❌ Unexpected error: {e}"


def send_real_sms(to_phone, body, account_sid, auth_token, from_phone):
    """Send actual SMS via Twilio REST API (no SDK needed)."""
    import urllib.request, urllib.parse, base64
    try:
        url  = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        data = urllib.parse.urlencode({
            "To":   to_phone,
            "From": from_phone,
            "Body": body,
        }).encode("utf-8")
        credentials = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Basic {credentials}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            result = json.loads(resp.read())
            sid = result.get("sid", "")
            return True, f"✅ SMS sent! Twilio SID: {sid}"
    except urllib.error.HTTPError as e:
        import json
        err = json.loads(e.read()).get("message", str(e))
        return False, f"❌ Twilio error: {err}"
    except Exception as e:
        return False, f"❌ Unexpected error: {e}"

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictaMaint AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --neon-cyan: #00f5ff;
    --neon-orange: #ff6b00;
    --neon-green: #39ff14;
    --neon-purple: #bf5fff;
    --dark-bg: #0a0e1a;
    --card-bg: #0f1629;
    --border: #1e3a5f;
    --text-dim: #7a9cc4;
}

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--dark-bg);
    color: #c8d8e8;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1520 100%);
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 20px rgba(0,245,255,0.4);
    letter-spacing: 2px;
}

.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f5ff, #ff6b00, #39ff14);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 0.2rem;
    text-shadow: none;
}

.subtitle {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-dim);
    font-size: 0.95rem;
    letter-spacing: 3px;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #0f1629, #111d35);
    border: 1px solid var(--border);
    border-left: 3px solid var(--neon-cyan);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 0 15px rgba(0,245,255,0.05), inset 0 0 30px rgba(0,0,0,0.3);
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--neon-cyan);
    text-shadow: 0 0 10px rgba(0,245,255,0.5);
}

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.alert-critical {
    background: linear-gradient(135deg, #2a0a0a, #1a0505);
    border: 1px solid #ff3333;
    border-left: 4px solid #ff3333;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #ff8080;
    font-family: 'Share Tech Mono', monospace;
    animation: pulse-red 2s infinite;
}

.alert-warning {
    background: linear-gradient(135deg, #2a1a00, #1a1000);
    border: 1px solid var(--neon-orange);
    border-left: 4px solid var(--neon-orange);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #ffaa55;
    font-family: 'Share Tech Mono', monospace;
}

.alert-safe {
    background: linear-gradient(135deg, #0a2a0a, #051a05);
    border: 1px solid var(--neon-green);
    border-left: 4px solid var(--neon-green);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #88ff88;
    font-family: 'Share Tech Mono', monospace;
}

.alert-sms {
    background: linear-gradient(135deg, #1a0a2a, #100520);
    border: 1px solid var(--neon-purple);
    border-left: 4px solid var(--neon-purple);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #cc99ff;
    font-family: 'Share Tech Mono', monospace;
    margin: 0.5rem 0;
}

.section-header {
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon-orange);
    font-size: 0.8rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #003366, #004080) !important;
    color: var(--neon-cyan) !important;
    border: 1px solid var(--neon-cyan) !important;
    border-radius: 4px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--neon-cyan), #0099cc) !important;
    color: #000 !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.4) !important;
}

.stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
    background: var(--card-bg) !important;
    color: #c8d8e8 !important;
    border-color: var(--border) !important;
}

.stSidebar {
    background: linear-gradient(180deg, #080c18, #0a1020) !important;
    border-right: 1px solid var(--border) !important;
}

.stSidebar .stMarkdown h3 {
    color: var(--neon-orange) !important;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 5px rgba(255,51,51,0.3); }
    50% { box-shadow: 0 0 20px rgba(255,51,51,0.6); }
}

@keyframes pulse-purple {
    0%, 100% { box-shadow: 0 0 5px rgba(191,95,255,0.3); }
    50% { box-shadow: 0 0 20px rgba(191,95,255,0.6); }
}

div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}

div[data-testid="stMetric"] label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--neon-cyan) !important;
}

.fps-bar-container {
    background: #0a0e1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 22px;
    overflow: hidden;
    margin: 0.3rem 0;
}

.rt-sensor-card {
    background: linear-gradient(135deg, #0f1629, #111d35);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    font-family: 'Share Tech Mono', monospace;
}

.log-entry {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    padding: 0.3rem 0.6rem;
    border-left: 2px solid var(--neon-orange);
    margin: 0.2rem 0;
    background: rgba(255,107,0,0.05);
    color: #c8d8e8;
}
</style>
""", unsafe_allow_html=True)

# ─── Utility Functions ──────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n_samples=5000):
    np.random.seed(42)
    equipment_types = ['Pump', 'Compressor', 'Motor', 'Turbine', 'Generator']
    manufacturers  = ['SiemensTech', 'ABB Industrial', 'GE Power', 'Honeywell', 'Bosch Rexroth']
    equipment_type = np.random.choice(equipment_types, n_samples)
    manufacturer   = np.random.choice(manufacturers, n_samples)
    age_years          = np.random.uniform(0.5, 20, n_samples)
    operating_hours    = age_years * np.random.uniform(1500, 2500, n_samples)
    temperature_c      = np.random.normal(75, 15, n_samples).clip(30, 150)
    vibration_mm_s     = np.random.exponential(2.5, n_samples).clip(0.1, 25)
    pressure_bar       = np.random.normal(8, 2, n_samples).clip(2, 20)
    current_amp        = np.random.normal(45, 10, n_samples).clip(10, 120)
    voltage_v          = np.random.normal(400, 20, n_samples).clip(350, 450)
    oil_viscosity      = np.random.normal(68, 8, n_samples).clip(40, 100)
    humidity_pct       = np.random.uniform(20, 90, n_samples)
    load_factor        = np.random.uniform(0.3, 1.0, n_samples)
    maintenance_count  = np.random.poisson(age_years * 0.8, n_samples)
    last_maintenance_d = np.random.uniform(1, 365, n_samples)
    rpm                = np.random.normal(1500, 200, n_samples).clip(500, 3600)
    power_kw         = (current_amp * voltage_v * 0.85) / 1000
    thermal_stress   = temperature_c * load_factor
    vibration_energy = vibration_mm_s ** 2
    base_rul = 365 - (age_years * 12) - (vibration_mm_s * 8) - (temperature_c * 0.5) \
               + (maintenance_count * 5) - (last_maintenance_d * 0.3) \
               + (oil_viscosity * 0.8) - (load_factor * 40) \
               + np.random.normal(0, 15, n_samples)
    rul_days = base_rul.clip(5, 500)
    df = pd.DataFrame({
        'equipment_type':     equipment_type,
        'manufacturer':       manufacturer,
        'age_years':          age_years.round(2),
        'operating_hours':    operating_hours.round(0).astype(int),
        'temperature_c':      temperature_c.round(1),
        'vibration_mm_s':     vibration_mm_s.round(3),
        'pressure_bar':       pressure_bar.round(2),
        'current_amp':        current_amp.round(1),
        'voltage_v':          voltage_v.round(1),
        'oil_viscosity':      oil_viscosity.round(1),
        'humidity_pct':       humidity_pct.round(1),
        'load_factor':        load_factor.round(3),
        'maintenance_count':  maintenance_count,
        'last_maintenance_days': last_maintenance_d.round(0).astype(int),
        'rpm':                rpm.round(0).astype(int),
        'power_kw':           power_kw.round(2),
        'thermal_stress':     thermal_stress.round(2),
        'vibration_energy':   vibration_energy.round(4),
        'rul_days':           rul_days.round(0).astype(int)
    })
    return df


@st.cache_resource
def train_model(df):
    le_type = LabelEncoder()
    le_mfr  = LabelEncoder()
    df = df.copy()
    df['equipment_type_enc'] = le_type.fit_transform(df['equipment_type'])
    df['manufacturer_enc']   = le_mfr.fit_transform(df['manufacturer'])
    feature_cols = [
        'equipment_type_enc', 'manufacturer_enc', 'age_years', 'operating_hours',
        'temperature_c', 'vibration_mm_s', 'pressure_bar', 'current_amp',
        'voltage_v', 'oil_viscosity', 'humidity_pct', 'load_factor',
        'maintenance_count', 'last_maintenance_days', 'rpm',
        'power_kw', 'thermal_stress', 'vibration_energy'
    ]
    X = df[feature_cols]
    y = df['rul_days']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.08,
        max_depth=5, subsample=0.85,
        min_samples_split=10, random_state=42
    )
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    metrics = {
        'MAE':  round(mean_absolute_error(y_test, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'R2':   round(r2_score(y_test, y_pred), 4),
    }
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return model, scaler, le_type, le_mfr, feature_cols, metrics, importances, X_test_sc, y_test, y_pred


def predict_rul(model, scaler, le_type, le_mfr, feature_cols, inputs: dict) -> float:
    row = pd.DataFrame([inputs])
    row['equipment_type_enc'] = le_type.transform(row['equipment_type'])
    row['manufacturer_enc']   = le_mfr.transform(row['manufacturer'])
    row = row[feature_cols]
    row_sc = scaler.transform(row)
    return float(model.predict(row_sc)[0])


def rul_status(rul):
    if rul < 30:
        return "CRITICAL", "#ff3333", "⛔"
    elif rul < 90:
        return "WARNING", "#ff6b00", "⚠️"
    else:
        return "HEALTHY", "#39ff14", "✅"


def compute_failure_probability(rul, temperature, vibration, load_factor, age_years, last_maintenance_days):
    """
    Compute Failure Probability Score (FPS) from 0–100.
    Uses a multi-factor sigmoid model:
    - RUL proximity to 0 increases probability
    - High temperature, vibration, load push it up
    - Old age and long gap since maintenance add risk
    Returns: fps (float 0-100), contributing factors dict
    """
    # Normalize each risk signal to 0–1
    rul_risk    = max(0, min(1, 1 - (rul / 365)))            # 0 days RUL = full risk
    temp_risk   = max(0, min(1, (temperature - 50) / 100))   # risk starts at 50°C
    vib_risk    = max(0, min(1, vibration / 20))              # risk maxes at 20 mm/s
    load_risk   = max(0, min(1, (load_factor - 0.5) / 0.5))  # risk above 50% load
    age_risk    = max(0, min(1, age_years / 20))              # 20yr = max age risk
    maint_risk  = max(0, min(1, last_maintenance_days / 365)) # 1yr gap = max risk

    # Weighted composite (weights sum to 1)
    weights = {
        'RUL Proximity':        (rul_risk,   0.35),
        'Temperature Stress':   (temp_risk,  0.20),
        'Vibration Level':      (vib_risk,   0.20),
        'Load Factor':          (load_risk,  0.10),
        'Equipment Age':        (age_risk,   0.10),
        'Maintenance Gap':      (maint_risk, 0.05),
    }
    raw_score = sum(v * w for v, w in weights.values())
    # Sigmoid sharpening — amplifies mid-range values
    fps = 100 / (1 + np.exp(-10 * (raw_score - 0.5)))
    contributions = {k: round(v * w * 100, 1) for k, (v, w) in weights.items()}
    return round(float(fps), 1), contributions


def simulate_realtime_sensors(base_inputs, noise_level=0.03):
    """Add small random jitter to simulate live sensor stream."""
    sensors = {}
    for k, v in base_inputs.items():
        if isinstance(v, float) or isinstance(v, int):
            if k in ('equipment_type_enc', 'manufacturer_enc', 'maintenance_count',
                     'operating_hours', 'last_maintenance_days'):
                sensors[k] = v
            else:
                sensors[k] = float(v) * (1 + np.random.uniform(-noise_level, noise_level))
        else:
            sensors[k] = v
    return sensors


def format_alert_message(equip_type, fps, rul, status, manufacturer):
    """Format SMS/Email alert body."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'subject': f"[PredictaMaint] {status} ALERT — {equip_type} ({manufacturer})",
        'body': f"""
=== PREDICTAMAINT AI ALERT ===
Timestamp  : {now}
Equipment  : {equip_type} by {manufacturer}
Status     : {status}
RUL        : {rul:.0f} days remaining
Failure Prob: {fps:.1f}%

{'⛔ IMMEDIATE ACTION REQUIRED — Schedule maintenance within 7 days.' if status == 'CRITICAL' else '⚠️ MAINTENANCE RECOMMENDED — Plan service within 30–90 days.'}

Sensor data logged. Log in to PredictaMaint dashboard for full details.
— PredictaMaint AI v2.5.0
""".strip()
    }


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ PREDICTAMAINT AI")
    st.markdown("<p style='font-family:Share Tech Mono;font-size:0.7rem;color:#7a9cc4;letter-spacing:2px'>INDUSTRIAL INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard", "🔮 Predict RUL", "📡 Real-Time Monitor",
         "📊 Data Explorer", "🧠 Model Insights", "📥 Dataset Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<p class='section-header'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    st.success("✅ Model: ONLINE")
    st.info("📡 Data Feed: ACTIVE")
    st.markdown("<p style='font-family:Share Tech Mono;font-size:0.65rem;color:#7a9cc4'>v2.5.0 | GradientBoosting + FPS</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p class='section-header'>📧 EMAIL CREDENTIALS</p>", unsafe_allow_html=True)
    st.markdown("""<p style='font-family:Share Tech Mono;font-size:0.65rem;color:#7a9cc4'>
    Gmail SMTP. Use an App Password (not your login).<br>
    <a href='https://myaccount.google.com/apppasswords' target='_blank' style='color:#ff6b00'>→ Get App Password</a></p>""", unsafe_allow_html=True)
    sender_email    = st.text_input("Sender Gmail", placeholder="yourapp@gmail.com", key="s_email")
    sender_password = st.text_input("Gmail App Password", type="password", placeholder="xxxx xxxx xxxx xxxx", key="s_pass")

    st.markdown("---")
    st.markdown("<p class='section-header'>📱 TWILIO SMS CREDENTIALS</p>", unsafe_allow_html=True)
    st.markdown("""<p style='font-family:Share Tech Mono;font-size:0.65rem;color:#7a9cc4'>
    Free trial at twilio.com.<br>
    <a href='https://console.twilio.com' target='_blank' style='color:#ff6b00'>→ Twilio Console</a></p>""", unsafe_allow_html=True)
    twilio_sid   = st.text_input("Account SID", placeholder="ACxxxxxxxxxxxxxxxx", key="t_sid")
    twilio_token = st.text_input("Auth Token", type="password", placeholder="your_auth_token", key="t_token")
    twilio_from  = st.text_input("Twilio From Number", placeholder="+14155550000", key="t_from")

# ─── Load Data + Train ─────────────────────────────────────────────────────────
df = generate_dataset(5000)
model, scaler, le_type, le_mfr, feature_cols, metrics, importances, X_test_sc, y_test, y_pred = train_model(df)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚙ PREDICTAMAINT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">[ AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR INDUSTRIAL EQUIPMENT ]</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<p class="section-header">// SYSTEM OVERVIEW</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equipment", f"{len(df):,}", "5 Types")
    c2.metric("Avg RUL", f"{df['rul_days'].mean():.0f} days", f"±{df['rul_days'].std():.0f}")
    c3.metric("Model R² Score", f"{metrics['R2']:.4f}", "Gradient Boost")
    c4.metric("Critical Units", f"{(df['rul_days']<30).sum()}", "RUL < 30 days")

    st.markdown("---")
    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        st.markdown('<p class="section-header">// RUL DISTRIBUTION BY EQUIPMENT TYPE</p>', unsafe_allow_html=True)
        fig = px.box(df, x='equipment_type', y='rul_days', color='equipment_type',
                     color_discrete_sequence=['#00f5ff','#ff6b00','#39ff14','#ff00ff','#ffff00'],
                     template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,14,26,0.8)',
            font_family='Share Tech Mono', showlegend=False,
            xaxis_title="Equipment Type", yaxis_title="Remaining Useful Life (days)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">// FLEET HEALTH STATUS</p>', unsafe_allow_html=True)
        critical = (df['rul_days'] < 30).sum()
        warning  = ((df['rul_days'] >= 30) & (df['rul_days'] < 90)).sum()
        healthy  = (df['rul_days'] >= 90).sum()
        fig_pie = go.Figure(go.Pie(
            labels=['CRITICAL', 'WARNING', 'HEALTHY'],
            values=[critical, warning, healthy],
            hole=0.55,
            marker_colors=['#ff3333','#ff6b00','#39ff14'],
            textinfo='label+percent',
            textfont_family='Share Tech Mono'
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Share Tech Mono', font_color='#c8d8e8',
            showlegend=False,
            annotations=[dict(text=f'{len(df)}<br>Units', x=0.5, y=0.5,
                              font_size=16, font_color='#00f5ff',
                              font_family='Orbitron', showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<p class="section-header">// SENSOR CORRELATION HEATMAP</p>', unsafe_allow_html=True)
    num_cols = ['age_years','temperature_c','vibration_mm_s','pressure_bar',
                'current_amp','load_factor','maintenance_count','last_maintenance_days','rul_days']
    corr = df[num_cols].corr()
    fig_heat = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         template='plotly_dark', text_auto='.2f')
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           font_family='Share Tech Mono', font_color='#c8d8e8')
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT RUL  (+ FPS + Alerts)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict RUL":
    st.markdown('<p class="section-header">// REMAINING USEFUL LIFE PREDICTOR + FAILURE PROBABILITY SCORE</p>', unsafe_allow_html=True)
    st.markdown("Enter real-time sensor readings to predict RUL, Failure Probability Score, and trigger alerts.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🏭 Equipment Info**")
        equip_type   = st.selectbox("Equipment Type", sorted(df['equipment_type'].unique()))
        manufacturer = st.selectbox("Manufacturer",   sorted(df['manufacturer'].unique()))
        age_years    = st.slider("Age (years)", 0.5, 20.0, 5.0, 0.1)
        op_hours     = st.number_input("Operating Hours", 500, 50000, int(age_years*2000), 100)
        maint_count  = st.number_input("Total Maintenance Count", 0, 50, int(age_years*0.8))
        last_maint   = st.slider("Days Since Last Maintenance", 1, 365, 90)

    with col_b:
        st.markdown("**🌡️ Thermal & Mechanical**")
        temperature = st.slider("Temperature (°C)", 30.0, 150.0, 75.0, 0.5)
        vibration   = st.slider("Vibration (mm/s)", 0.1, 25.0, 2.5, 0.1)
        pressure    = st.slider("Pressure (bar)", 2.0, 20.0, 8.0, 0.1)
        rpm         = st.slider("RPM", 500, 3600, 1500, 10)
        load_factor = st.slider("Load Factor", 0.3, 1.0, 0.7, 0.01)

    with col_c:
        st.markdown("**⚡ Electrical & Fluid**")
        current    = st.slider("Current (A)", 10.0, 120.0, 45.0, 0.5)
        voltage    = st.slider("Voltage (V)", 350.0, 450.0, 400.0, 1.0)
        oil_visc   = st.slider("Oil Viscosity (cSt)", 40.0, 100.0, 68.0, 0.5)
        humidity   = st.slider("Humidity (%)", 20.0, 90.0, 50.0, 1.0)

    # ── Alert Config ──
    st.markdown("---")
    st.markdown('<p class="section-header">// ALERT CONFIGURATION (SMS / EMAIL)</p>', unsafe_allow_html=True)
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    with alert_col1:
        alert_email = st.text_input("📧 Email Address", placeholder="engineer@plant.com")
    with alert_col2:
        alert_phone = st.text_input("📱 Phone (SMS)", placeholder="+91 98765 43210")
    with alert_col3:
        alert_threshold = st.selectbox("🔔 Alert On", ["CRITICAL only", "WARNING + CRITICAL", "All predictions"])
    st.markdown("---")

    if st.button("🔮 PREDICT REMAINING USEFUL LIFE + FAILURE SCORE", use_container_width=True):
        power_kw       = (current * voltage * 0.85) / 1000
        thermal_stress = temperature * load_factor
        vib_energy     = vibration ** 2

        inputs = {
            'equipment_type': equip_type,
            'manufacturer':   manufacturer,
            'age_years':      age_years,
            'operating_hours': op_hours,
            'temperature_c':  temperature,
            'vibration_mm_s': vibration,
            'pressure_bar':   pressure,
            'current_amp':    current,
            'voltage_v':      voltage,
            'oil_viscosity':  oil_visc,
            'humidity_pct':   humidity,
            'load_factor':    load_factor,
            'maintenance_count': maint_count,
            'last_maintenance_days': last_maint,
            'rpm':            rpm,
            'power_kw':       power_kw,
            'thermal_stress': thermal_stress,
            'vibration_energy': vib_energy,
        }

        rul = predict_rul(model, scaler, le_type, le_mfr, feature_cols, inputs)
        rul = max(5, min(500, rul))
        status, color, icon = rul_status(rul)

        # Failure Probability Score
        fps, contributions = compute_failure_probability(
            rul, temperature, vibration, load_factor, age_years, last_maint
        )
        fps_color = "#ff3333" if fps >= 70 else ("#ff6b00" if fps >= 40 else "#39ff14")

        # ── RUL Display ──
        res_col, fps_col = st.columns(2)

        with res_col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0f1629,#111d35);
                        border:2px solid {color};border-radius:12px;
                        padding:2rem;text-align:center;margin:1.5rem 0;
                        box-shadow:0 0 30px {color}33">
                <div style="font-family:'Share Tech Mono';font-size:0.85rem;
                            color:#7a9cc4;letter-spacing:4px;margin-bottom:0.5rem">
                    PREDICTED REMAINING USEFUL LIFE
                </div>
                <div style="font-family:'Orbitron',monospace;font-size:4rem;
                            font-weight:900;color:{color};
                            text-shadow:0 0 30px {color}88;line-height:1">
                    {rul:.0f}
                </div>
                <div style="font-family:'Share Tech Mono';font-size:1rem;
                            color:{color};letter-spacing:3px">DAYS</div>
                <div style="font-family:'Orbitron';font-size:1.3rem;
                            color:{color};margin-top:1rem;letter-spacing:4px">
                    {icon} STATUS: {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── FPS Display ──
        with fps_col:
            bar_width = int(fps)
            bar_color = fps_color
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0f1629,#111d35);
                        border:2px solid {fps_color};border-radius:12px;
                        padding:2rem;text-align:center;margin:1.5rem 0;
                        box-shadow:0 0 30px {fps_color}33">
                <div style="font-family:'Share Tech Mono';font-size:0.85rem;
                            color:#7a9cc4;letter-spacing:4px;margin-bottom:0.5rem">
                    FAILURE PROBABILITY SCORE
                </div>
                <div style="font-family:'Orbitron',monospace;font-size:4rem;
                            font-weight:900;color:{fps_color};
                            text-shadow:0 0 30px {fps_color}88;line-height:1">
                    {fps:.1f}
                </div>
                <div style="font-family:'Share Tech Mono';font-size:1rem;
                            color:{fps_color};letter-spacing:3px">% FAILURE RISK</div>
                <div style="background:#0a0e1a;border-radius:6px;height:14px;
                            margin-top:1rem;overflow:hidden;border:1px solid #1e3a5f">
                    <div style="width:{bar_width}%;height:100%;
                                background:linear-gradient(90deg,#39ff14,{fps_color});
                                transition:width 1s ease;border-radius:6px"></div>
                </div>
                <div style="font-family:'Share Tech Mono';font-size:0.8rem;
                            color:{fps_color};margin-top:0.5rem">
                    {'🔴 HIGH RISK' if fps >= 70 else ('🟠 MODERATE RISK' if fps >= 40 else '🟢 LOW RISK')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── FPS Factor Breakdown ──
        st.markdown('<p class="section-header">// FAILURE PROBABILITY FACTOR BREAKDOWN</p>', unsafe_allow_html=True)
        factor_cols = st.columns(len(contributions))
        for i, (factor, contrib) in enumerate(contributions.items()):
            c_color = "#ff3333" if contrib > 15 else ("#ff6b00" if contrib > 8 else "#39ff14")
            factor_cols[i].markdown(f"""
            <div style="background:#0f1629;border:1px solid #1e3a5f;border-top:3px solid {c_color};
                        border-radius:8px;padding:0.8rem;text-align:center">
                <div style="font-family:'Share Tech Mono';font-size:0.65rem;
                            color:#7a9cc4;letter-spacing:1px">{factor}</div>
                <div style="font-family:'Orbitron';font-size:1.4rem;font-weight:700;
                            color:{c_color}">{contrib}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Maintenance Recommendations ──
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Maintenance Date", f"In {rul:.0f} days")
        r2.metric("Risk Level", status)
        r3.metric("Power Draw", f"{power_kw:.1f} kW")

        if status == "CRITICAL":
            st.markdown('<div class="alert-critical">⛔ IMMEDIATE ACTION REQUIRED — Schedule maintenance within 7 days. High failure probability detected. Reduce load factor and monitor vibration continuously.</div>', unsafe_allow_html=True)
        elif status == "WARNING":
            st.markdown('<div class="alert-warning">⚠️ MAINTENANCE RECOMMENDED — Plan service within 30–90 days. Monitor temperature and vibration trends closely. Review oil quality.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-safe">✅ EQUIPMENT HEALTHY — Continue normal operations. Schedule next routine inspection per standard maintenance calendar.</div>', unsafe_allow_html=True)

        # ── Real Alert Sending ──
        st.markdown("---")
        st.markdown('<p class="section-header">// ALERT NOTIFICATIONS</p>', unsafe_allow_html=True)

        should_alert = (
            alert_threshold == "All predictions" or
            (alert_threshold == "WARNING + CRITICAL" and status in ("WARNING", "CRITICAL")) or
            (alert_threshold == "CRITICAL only" and status == "CRITICAL")
        )

        alert_msg = format_alert_message(equip_type, fps, rul, status, manufacturer)

        if should_alert:
            sent_any = False

            # ── Real Email ──
            if alert_email:
                sent_any = True
                if sender_email and sender_password:
                    with st.spinner("📧 Sending email..."):
                        ok, result_msg = send_real_email(
                            to_email=alert_email,
                            subject=alert_msg['subject'],
                            body=alert_msg['body'],
                            sender_email=sender_email,
                            sender_password=sender_password,
                        )
                    if ok:
                        st.markdown(f"""
                        <div class="alert-sms" style="animation:pulse-purple 2s infinite">
                            📧 <b>EMAIL SENT</b> → {alert_email}<br>
                            <span style="color:#7a9cc4">Subject: {alert_msg['subject']}</span><br>
                            <span style="color:#39ff14">{result_msg}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("📄 View Email Body"):
                            st.code(alert_msg['body'], language=None)
                    else:
                        st.markdown(f"""
                        <div class="alert-critical">
                            📧 Email to {alert_email} failed.<br>{result_msg}
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("💡 Make sure you entered a Gmail App Password (not your Gmail login). Enable 2FA then visit myaccount.google.com/apppasswords")
                else:
                    st.warning("⚠️ Enter your Sender Gmail + App Password in the sidebar to actually send email.")
                    with st.expander("📄 Preview Email (not sent)"):
                        st.code(alert_msg['body'], language=None)

            # ── Real SMS ──
            if alert_phone:
                sent_any = True
                sms_body = f"[PredictaMaint] {status} — {equip_type} ({manufacturer}). RUL={rul:.0f}d, Failure Risk={fps:.1f}%. {'IMMEDIATE ACTION REQUIRED.' if status=='CRITICAL' else 'Plan maintenance soon.'}"
                if twilio_sid and twilio_token and twilio_from:
                    with st.spinner("📱 Sending SMS..."):
                        ok, result_msg = send_real_sms(
                            to_phone=alert_phone,
                            body=sms_body,
                            account_sid=twilio_sid,
                            auth_token=twilio_token,
                            from_phone=twilio_from,
                        )
                    if ok:
                        st.markdown(f"""
                        <div class="alert-sms">
                            📱 <b>SMS SENT</b> → {alert_phone}<br>
                            <span style="color:#c8d8e8;font-size:0.8rem">{sms_body}</span><br>
                            <span style="color:#39ff14">{result_msg}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-critical">
                            📱 SMS to {alert_phone} failed.<br>{result_msg}
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("💡 Make sure your Twilio number is verified and the destination number is verified on free trial. Phone format: +919876543210")
                else:
                    st.warning("⚠️ Enter your Twilio credentials in the sidebar to actually send SMS.")
                    st.code(f"Preview SMS: {sms_body}", language=None)

            if not sent_any:
                st.info("ℹ️ Enter an email or phone number above to receive alerts.")
        else:
            st.markdown(f'<div class="alert-safe">🔕 No alert triggered — equipment status is <b>{status}</b>, below your alert threshold ({alert_threshold}).</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — REAL-TIME DASHBOARD (NEW)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Real-Time Monitor":
    st.markdown('<p class="section-header">// REAL-TIME SENSOR MONITORING DASHBOARD</p>', unsafe_allow_html=True)
    st.markdown("Live sensor simulation with auto-refresh. Select an equipment profile and watch the dashboard update.")

    # Equipment selection
    rt_col1, rt_col2, rt_col3, rt_col4 = st.columns(4)
    with rt_col1:
        rt_equip = st.selectbox("Equipment", sorted(df['equipment_type'].unique()), key="rt_equip")
    with rt_col2:
        rt_mfr = st.selectbox("Manufacturer", sorted(df['manufacturer'].unique()), key="rt_mfr")
    with rt_col3:
        rt_age = st.slider("Age (yrs)", 0.5, 20.0, 5.0, 0.5, key="rt_age")
    with rt_col4:
        refresh_rate = st.selectbox("Refresh (sec)", [3, 5, 10, 30], index=1)

    auto_refresh = st.toggle("🔄 Enable Auto-Refresh", value=False)
    st.markdown("---")

    # Build a base input for the selected equipment
    base = {
        'equipment_type': rt_equip,
        'manufacturer':   rt_mfr,
        'age_years':      rt_age,
        'operating_hours': int(rt_age * 2000),
        'temperature_c':  75.0,
        'vibration_mm_s': 2.5,
        'pressure_bar':   8.0,
        'current_amp':    45.0,
        'voltage_v':      400.0,
        'oil_viscosity':  68.0,
        'humidity_pct':   50.0,
        'load_factor':    0.7,
        'maintenance_count': int(rt_age * 0.8),
        'last_maintenance_days': 90,
        'rpm':            1500.0,
        'power_kw':       (45.0 * 400.0 * 0.85) / 1000,
        'thermal_stress': 75.0 * 0.7,
        'vibration_energy': 2.5 ** 2,
    }

    # Simulate live readings
    live = simulate_realtime_sensors(base, noise_level=0.05)
    live['power_kw']       = (live['current_amp'] * live['voltage_v'] * 0.85) / 1000
    live['thermal_stress'] = live['temperature_c'] * live['load_factor']
    live['vibration_energy'] = live['vibration_mm_s'] ** 2

    rul_live = predict_rul(model, scaler, le_type, le_mfr, feature_cols, live)
    rul_live = max(5, min(500, rul_live))
    status_live, color_live, icon_live = rul_status(rul_live)
    fps_live, _ = compute_failure_probability(
        rul_live, live['temperature_c'], live['vibration_mm_s'],
        live['load_factor'], rt_age, live['last_maintenance_days']
    )
    fps_color_live = "#ff3333" if fps_live >= 70 else ("#ff6b00" if fps_live >= 40 else "#39ff14")

    # ── Live KPI Row ──
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🕐 Last Updated", now_str)
    k2.metric("🔋 RUL", f"{rul_live:.0f} days")
    k3.metric("💀 Failure Risk", f"{fps_live:.1f}%")
    k4.metric("🌡️ Temperature", f"{live['temperature_c']:.1f} °C")
    k5.metric("📳 Vibration", f"{live['vibration_mm_s']:.2f} mm/s")

    st.markdown("---")

    # ── Live Gauge Charts ──
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

    def make_gauge(value, max_val, title, color, unit=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            delta={'reference': max_val * 0.5, 'increasing': {'color': "#ff3333"}, 'decreasing': {'color': "#39ff14"}},
            number={'suffix': unit, 'font': {'family': 'Orbitron', 'color': color, 'size': 28}},
            title={'text': title, 'font': {'family': 'Share Tech Mono', 'color': '#7a9cc4', 'size': 12}},
            gauge={
                'axis': {'range': [0, max_val], 'tickcolor': '#1e3a5f'},
                'bar': {'color': color},
                'bgcolor': '#0a0e1a',
                'bordercolor': '#1e3a5f',
                'steps': [
                    {'range': [0, max_val*0.5], 'color': '#0a2a0a'},
                    {'range': [max_val*0.5, max_val*0.75], 'color': '#2a1a00'},
                    {'range': [max_val*0.75, max_val], 'color': '#2a0a0a'},
                ],
                'threshold': {'line': {'color': '#ff3333', 'width': 3}, 'value': max_val * 0.8}
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#c8d8e8',
            height=220, margin=dict(l=20, r=20, t=40, b=10)
        )
        return fig

    with gauge_col1:
        st.plotly_chart(make_gauge(live['temperature_c'], 150, "TEMPERATURE (°C)", "#ff6b00", "°C"), use_container_width=True)
    with gauge_col2:
        st.plotly_chart(make_gauge(live['vibration_mm_s'], 25, "VIBRATION (mm/s)", "#00f5ff", " mm/s"), use_container_width=True)
    with gauge_col3:
        st.plotly_chart(make_gauge(live['load_factor'] * 100, 100, "LOAD FACTOR (%)", "#39ff14", "%"), use_container_width=True)

    # ── Sensor Values Table ──
    st.markdown('<p class="section-header">// LIVE SENSOR READINGS</p>', unsafe_allow_html=True)
    sensor_display = {
        "Pressure (bar)":     f"{live['pressure_bar']:.2f}",
        "Current (A)":        f"{live['current_amp']:.1f}",
        "Voltage (V)":        f"{live['voltage_v']:.1f}",
        "Power (kW)":         f"{live['power_kw']:.2f}",
        "Oil Viscosity (cSt)":f"{live['oil_viscosity']:.1f}",
        "Humidity (%)":       f"{live['humidity_pct']:.1f}",
        "RPM":                f"{live['rpm']:.0f}",
        "Thermal Stress":     f"{live['thermal_stress']:.1f}",
    }
    sens_cols = st.columns(4)
    for i, (label, val) in enumerate(sensor_display.items()):
        with sens_cols[i % 4]:
            st.markdown(f"""
            <div class="rt-sensor-card">
                <div style="font-size:0.7rem;color:#7a9cc4;letter-spacing:2px">{label}</div>
                <div style="font-family:'Orbitron';font-size:1.4rem;color:#00f5ff">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Status Banner ──
    st.markdown("---")
    if status_live == "CRITICAL":
        st.markdown(f'<div class="alert-critical">⛔ LIVE STATUS: CRITICAL — RUL {rul_live:.0f} days | Failure Risk {fps_live:.1f}% — Immediate action required.</div>', unsafe_allow_html=True)
    elif status_live == "WARNING":
        st.markdown(f'<div class="alert-warning">⚠️ LIVE STATUS: WARNING — RUL {rul_live:.0f} days | Failure Risk {fps_live:.1f}% — Schedule maintenance soon.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-safe">✅ LIVE STATUS: HEALTHY — RUL {rul_live:.0f} days | Failure Risk {fps_live:.1f}% — Operating normally.</div>', unsafe_allow_html=True)

    # ── Auto Refresh ──
    if auto_refresh:
        st.info(f"🔄 Auto-refreshing every {refresh_rate} seconds... (toggle off to stop)")
        time.sleep(refresh_rate)
        st.rerun()
    else:
        if st.button("🔄 REFRESH NOW", use_container_width=True):
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<p class="section-header">// SENSOR DATA EXPLORER</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        equip_filter = st.multiselect("Filter by Equipment Type",
                                       df['equipment_type'].unique(),
                                       default=list(df['equipment_type'].unique()))
    with c2:
        x_axis = st.selectbox("X-Axis Feature", ['age_years','temperature_c','vibration_mm_s',
                                                    'pressure_bar','load_factor','operating_hours'])
    y_axis = 'rul_days'

    filtered = df[df['equipment_type'].isin(equip_filter)]

    fig_scatter = px.scatter(filtered, x=x_axis, y=y_axis,
                             color='equipment_type', opacity=0.6, size_max=6,
                             color_discrete_sequence=['#00f5ff','#ff6b00','#39ff14','#ff00ff','#ffff00'],
                             template='plotly_dark',
                             labels={x_axis: x_axis.replace('_',' ').title(),
                                     y_axis: 'Remaining Useful Life (days)'})
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(10,14,26,0.8)',
                               font_family='Share Tech Mono')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<p class="section-header">// RAW DATA SAMPLE</p>', unsafe_allow_html=True)
    st.dataframe(
        filtered.sample(min(200, len(filtered))).reset_index(drop=True),
        use_container_width=True, height=350
    )

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ DOWNLOAD FILTERED DATA (CSV)", csv,
                       "predictive_maintenance_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.markdown('<p class="section-header">// MODEL PERFORMANCE & INSIGHTS</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error", f"{metrics['MAE']} days")
    m2.metric("RMSE", f"{metrics['RMSE']} days")
    m3.metric("R² Score", f"{metrics['R2']}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-header">// ACTUAL vs PREDICTED</p>', unsafe_allow_html=True)
        y_test_arr = np.array(y_test)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test_arr[:300], y=y_pred[:300],
            mode='markers', marker=dict(color='#00f5ff', opacity=0.5, size=5),
            name='Predictions'
        ))
        mn, mx = y_test_arr.min(), y_test_arr.max()
        fig_pred.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx],
            mode='lines', line=dict(color='#ff6b00', dash='dash', width=2),
            name='Perfect Fit'))
        fig_pred.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,14,26,0.8)',
            font_family='Share Tech Mono', font_color='#c8d8e8',
            xaxis_title='Actual RUL (days)', yaxis_title='Predicted RUL (days)'
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">// FEATURE IMPORTANCE (TOP 12)</p>', unsafe_allow_html=True)
        top12 = importances.head(12)
        fig_imp = go.Figure(go.Bar(
            x=top12.values, y=top12.index,
            orientation='h',
            marker=dict(
                color=top12.values,
                colorscale=[[0,'#003366'],[0.5,'#0099cc'],[1,'#00f5ff']],
                showscale=False
            )
        ))
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,14,26,0.8)',
            font_family='Share Tech Mono', font_color='#c8d8e8',
            xaxis_title='Importance Score', yaxis_autorange='reversed'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<p class="section-header">// RESIDUAL DISTRIBUTION</p>', unsafe_allow_html=True)
    residuals = np.array(y_test) - y_pred
    fig_res = px.histogram(residuals, nbins=60, template='plotly_dark',
                           color_discrete_sequence=['#00f5ff'],
                           labels={'value':'Residual (days)', 'count':'Frequency'})
    fig_res.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(10,14,26,0.8)',
                          font_family='Share Tech Mono')
    st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATASET INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Dataset Info":
    st.markdown('<p class="section-header">// DATASET & PROJECT INFORMATION</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 📦 Dataset Details
    This project uses a **synthetically generated industrial equipment dataset** that closely mirrors
    real-world sensor readings from manufacturing environments.
    """)

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Samples", f"{len(df):,}")
    d2.metric("Features", f"{len(df.columns)-1}")
    d3.metric("Target Variable", "RUL (days)")

    st.markdown("---")
    st.markdown("### 🌐 Real-World Public Datasets You Can Use")

    datasets = [
        ("NASA CMAPSS Turbofan Engine Degradation",
         "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository",
         "Turbofan engine run-to-failure data. Gold standard for RUL prediction."),
        ("UCI ML — AI4I 2020 Predictive Maintenance",
         "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
         "10,000 data points with tool wear, temperature, torque sensors."),
        ("Kaggle — Predictive Maintenance Dataset AI4I",
         "https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
         "Same AI4I dataset, easy Kaggle download."),
        ("Microsoft Azure Predictive Maintenance",
         "https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance",
         "100 machines × 3 years of sensor data. Failures, errors, maintenance logs."),
        ("PRONOSTIA (FEMTO Bearing Dataset)",
         "https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge",
         "Bearing degradation data used in IEEE PHM 2012 challenge."),
    ]

    for name, url, desc in datasets:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-family:'Orbitron';font-size:0.85rem;color:#00f5ff">{name}</div>
            <div style="font-family:'Share Tech Mono';font-size:0.75rem;color:#7a9cc4;margin:0.3rem 0">{desc}</div>
            <a href="{url}" target="_blank" style="font-family:'Share Tech Mono';font-size:0.75rem;
               color:#ff6b00;text-decoration:none">🔗 {url}</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 Feature Descriptions")
    feat_df = pd.DataFrame({
        'Feature':     df.columns.tolist(),
        'Type':        ['Categorical']*2 + ['Numeric']*16 + ['Target'],
        'Description': [
            'Type of industrial equipment',
            'Equipment manufacturer',
            'Equipment age in years',
            'Total cumulative operating hours',
            'Operating temperature in Celsius',
            'Vibration level in mm/s',
            'System pressure in bar',
            'Electrical current draw in Amperes',
            'Supply voltage in Volts',
            'Lubrication oil viscosity (cSt)',
            'Ambient humidity percentage',
            'Current load as fraction of max capacity',
            'Total number of maintenance events',
            'Days since last maintenance',
            'Rotational speed in RPM',
            'Computed power draw in kW',
            'Temperature × Load Factor composite',
            'Squared vibration (energy proxy)',
            '🎯 Remaining Useful Life in days'
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
