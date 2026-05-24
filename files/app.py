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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

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
</style>
""", unsafe_allow_html=True)

# ─── Utility Functions ──────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n_samples=5000):
    """Generate realistic industrial equipment sensor dataset."""
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

    # Engineered features
    power_kw         = (current_amp * voltage_v * 0.85) / 1000
    thermal_stress   = temperature_c * load_factor
    vibration_energy = vibration_mm_s ** 2

    # Target: Remaining Useful Life (days) — realistic formula
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
    """Train GradientBoosting model and return model + scaler + encoders."""
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


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ PREDICTAMAINT AI")
    st.markdown("<p style='font-family:Share Tech Mono;font-size:0.7rem;color:#7a9cc4;letter-spacing:2px'>INDUSTRIAL INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard", "🔮 Predict RUL", "📊 Data Explorer", "🧠 Model Insights", "📥 Dataset Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<p class='section-header'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    st.success("✅ Model: ONLINE")
    st.info("📡 Data Feed: ACTIVE")
    st.markdown("<p style='font-family:Share Tech Mono;font-size:0.65rem;color:#7a9cc4'>v2.4.1 | GradientBoosting</p>", unsafe_allow_html=True)

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
# PAGE 2 — PREDICT RUL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict RUL":
    st.markdown('<p class="section-header">// REMAINING USEFUL LIFE PREDICTOR</p>', unsafe_allow_html=True)
    st.markdown("Enter real-time sensor readings to predict when the equipment needs maintenance.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🏭 Equipment Info**")
        equip_type  = st.selectbox("Equipment Type", sorted(df['equipment_type'].unique()))
        manufacturer = st.selectbox("Manufacturer",  sorted(df['manufacturer'].unique()))
        age_years   = st.slider("Age (years)", 0.5, 20.0, 5.0, 0.1)
        op_hours    = st.number_input("Operating Hours", 500, 50000, int(age_years*2000), 100)
        maint_count = st.number_input("Total Maintenance Count", 0, 50, int(age_years*0.8))
        last_maint  = st.slider("Days Since Last Maintenance", 1, 365, 90)

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

    st.markdown("---")

    if st.button("🔮 PREDICT REMAINING USEFUL LIFE", use_container_width=True):
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

        # Big result display
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

        # Maintenance recommendation
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
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
# PAGE 4 — MODEL INSIGHTS
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
# PAGE 5 — DATASET INFO
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
