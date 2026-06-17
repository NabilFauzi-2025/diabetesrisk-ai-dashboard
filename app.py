
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="DiabetesRisk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CSS DESIGN
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b0f19;
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        max-width: 1600px;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #253044;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #e5e7eb !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .profile-card {
        background: linear-gradient(135deg, #1a2030, #111827);
        border: 1px solid #263246;
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.22);
    }

    .avatar-circle {
        width: 46px;
        height: 46px;
        min-width: 46px;
        border-radius: 50%;
        background: linear-gradient(135deg, #fb3f73, #2dd4bf);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        box-shadow: 0 0 18px rgba(251, 63, 115, 0.30);
    }

    .profile-text {
        text-align: left;
    }

    .patient-name {
        color: #f8fafc;
        font-size: 14px;
        font-weight: 900;
        margin-bottom: 3px;
    }

    .patient-info {
        color: #8b98aa;
        font-size: 10.5px;
        line-height: 1.25;
    }

    .project-badge {
        display: inline-block;
        background: rgba(45, 212, 191, 0.12);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.45);
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 9.5px;
        font-weight: 800;
        margin-top: 5px;
    }

    .hero {
        background:
            radial-gradient(circle at 20% 20%, rgba(236, 72, 153, 0.25), transparent 28%),
            radial-gradient(circle at 85% 25%, rgba(20, 184, 166, 0.20), transparent 30%),
            linear-gradient(135deg, #111827 0%, #111827 55%, #13261f 100%);
        border: 1px solid #263246;
        border-radius: 24px;
        padding: 26px 34px;
        margin-bottom: 16px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.28);
    }

    .hero h1 {
        font-size: 46px;
        line-height: 1.03;
        margin: 0;
        font-weight: 900;
        color: #fb3f73;
        letter-spacing: -1px;
    }

    .hero p {
        color: #b6c2d1;
        font-size: 15px;
        margin-top: 12px;
        max-width: 900px;
    }

    .kpi-card {
        background: #1a2030;
        border: 1px solid #263246;
        border-radius: 16px;
        padding: 14px 16px;
        height: 118px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.20);
    }

    .kpi-label {
        color: #8b98aa;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 800;
    }

    .kpi-value {
        margin-top: 6px;
        color: #f8fafc;
        font-size: 23px;
        font-weight: 900;
    }

    .kpi-value-pink {
        margin-top: 6px;
        color: #fb3f73;
        font-size: 23px;
        font-weight: 900;
    }

    .kpi-value-teal {
        margin-top: 6px;
        color: #2dd4bf;
        font-size: 23px;
        font-weight: 900;
    }

    .kpi-desc {
        color: #9ca3af;
        font-size: 10.5px;
        line-height: 1.25;
        margin-top: 5px;
    }

    .panel-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .risk-pill-high {
        display: inline-block;
        background: rgba(251, 63, 115, 0.12);
        color: #fb3f73;
        border: 1px solid rgba(251, 63, 115, 0.55);
        border-radius: 999px;
        padding: 8px 18px;
        font-weight: 900;
        font-size: 13px;
        margin-top: 8px;
    }

    .risk-pill-medium {
        display: inline-block;
        background: rgba(251, 146, 60, 0.12);
        color: #fb923c;
        border: 1px solid rgba(251, 146, 60, 0.55);
        border-radius: 999px;
        padding: 8px 18px;
        font-weight: 900;
        font-size: 13px;
        margin-top: 8px;
    }

    .risk-pill-low {
        display: inline-block;
        background: rgba(45, 212, 191, 0.12);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.55);
        border-radius: 999px;
        padding: 8px 18px;
        font-weight: 900;
        font-size: 13px;
        margin-top: 8px;
    }

    .action-box {
        background: rgba(251, 63, 115, 0.08);
        border-left: 5px solid #fb3f73;
        border-radius: 14px;
        padding: 14px;
        color: #e5e7eb;
        font-size: 13px;
        line-height: 1.45;
    }

    .finding-box {
        background: rgba(45, 212, 191, 0.08);
        border-left: 5px solid #2dd4bf;
        border-radius: 14px;
        padding: 14px;
        color: #e5e7eb;
        font-size: 13px;
        line-height: 1.45;
        margin-top: 12px;
    }

    .detail-header {
        margin-top: 170px;
        margin-bottom: 12px;
        padding-top: 20px;
        border-top: 1px solid #263246;
        color: #f8fafc;
        font-size: 24px;
        font-weight: 900;
    }

    .footer-note {
        color: #8b98aa;
        font-size: 11px;
        margin-top: 5px;
    }

    div[data-testid="stSlider"] {
        padding-top: 0px;
        padding-bottom: 0px;
    }

    div[data-testid="stSelectbox"] {
        padding-top: 0px;
        padding-bottom: 0px;
        margin-bottom: -10px;
    }

    .stSlider label, .stSelectbox label {
        color: #e5e7eb !important;
        font-size: 12px !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    dataset = fetch_ucirepo(id=891)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    if "Diabetes_binary" not in df.columns:
        target_col = y.columns[0]
        df.rename(columns={target_col: "Diabetes_binary"}, inplace=True)

    if df["Diabetes_binary"].max() > 1:
        df["Diabetes_binary"] = (df["Diabetes_binary"] > 0).astype(int)

    selected_cols = [
        "Diabetes_binary", "HighBP", "HighChol", "BMI", "Smoker",
        "PhysActivity", "Fruits", "Veggies", "Age", "Education", "Income"
    ]

    return df[selected_cols].dropna()


df = load_data()


# =========================
# MODEL
# =========================
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    return model


model = train_model(df)

feature_cols = [
    "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity",
    "Fruits", "Veggies", "Age", "Education", "Income"
]

age_map = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+"
}

income_map = {
    1: "< $10k", 2: "$10k-$15k", 3: "$15k-$20k", 4: "$20k-$25k",
    5: "$25k-$35k", 6: "$35k-$50k", 7: "$50k-$75k", 8: "$75k+"
}


def yes_no_to_num(value):
    return 1 if value == "Yes" else 0


# =========================
# SIDEBAR PROFILE
# =========================
st.sidebar.markdown(
    """
    <div class="profile-card">
        <div class="avatar-circle">👨‍🦱</div>
        <div class="profile-text">
            <div class="patient-name">Ahmad bin Ali</div>
            <div class="patient-info">
                Demo patient • Male<br>
                Age group: 60–64<br>
                Screening candidate
            </div>
            <div class="project-badge">DiabetesRisk AI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("#### Blood & Body Markers")
bmi = st.sidebar.slider("BMI", 12, 60, 20)
high_bp = st.sidebar.selectbox("High Blood Pressure", ["Yes", "No"], index=0)
high_chol = st.sidebar.selectbox("High Cholesterol", ["Yes", "No"], index=0)

st.sidebar.markdown("#### Lifestyle")
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"], index=0)
phys_activity = st.sidebar.selectbox("Physical Activity", ["Yes", "No"], index=0)
fruits = st.sidebar.selectbox("Fruits", ["No", "Yes"], index=0)
veggies = st.sidebar.selectbox("Vegetables", ["No", "Yes"], index=0)

st.sidebar.markdown("#### Demographics")
age_label = st.sidebar.selectbox("Age Group", list(age_map.values()), index=8)
income_label = st.sidebar.selectbox("Income", list(income_map.values()), index=7)

age_code = list(age_map.keys())[list(age_map.values()).index(age_label)]
income_code = list(income_map.keys())[list(income_map.values()).index(income_label)]

input_data = pd.DataFrame({
    "HighBP": [yes_no_to_num(high_bp)],
    "HighChol": [yes_no_to_num(high_chol)],
    "BMI": [bmi],
    "Smoker": [yes_no_to_num(smoker)],
    "PhysActivity": [yes_no_to_num(phys_activity)],
    "Fruits": [yes_no_to_num(fruits)],
    "Veggies": [yes_no_to_num(veggies)],
    "Age": [age_code],
    "Education": [5],
    "Income": [income_code]
})

risk_probability = model.predict_proba(input_data)[0][1] * 100

if risk_probability >= 70:
    risk_level = "HIGH RISK"
    risk_class = "risk-pill-high"
elif risk_probability >= 40:
    risk_level = "MEDIUM RISK"
    risk_class = "risk-pill-medium"
else:
    risk_level = "LOW RISK"
    risk_class = "risk-pill-low"


# =========================
# FEATURE IMPACT
# =========================
scaler = model.named_steps["scaler"]
logreg = model.named_steps["logreg"]

scaled_input = scaler.transform(input_data[feature_cols])[0]
coef = logreg.coef_[0]
impact = scaled_input * coef

impact_df = pd.DataFrame({
    "Feature": feature_cols,
    "Impact": impact
})

display_names = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "BMI": "BMI",
    "Smoker": "Smoking",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Intake",
    "Veggies": "Vegetable Intake",
    "Age": "Age Group",
    "Education": "Education",
    "Income": "Income"
}

impact_df["Feature"] = impact_df["Feature"].map(display_names)
impact_df["AbsImpact"] = impact_df["Impact"].abs()
impact_df = impact_df.sort_values("AbsImpact", ascending=True).tail(8)

impact_colors = ["#fb3f73" if value > 0 else "#2dd4bf" for value in impact_df["Impact"]]


# =========================
# SUMMARY STATS
# =========================
total_records = df.shape[0]
diabetes_percent = df["Diabetes_binary"].mean() * 100

bp_risk = df.groupby("HighBP")["Diabetes_binary"].mean().mul(100).reindex([0, 1]).fillna(0)
bmi_summary = df.groupby("Diabetes_binary")["BMI"].mean().reindex([0, 1]).fillna(0)


# =========================
# HERO TITLE
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>DiabetesRisk AI<br>Predict Early. Prevent Smarter.</h1>
        <p>
        A data science decision-support dashboard that uses health, lifestyle, and demographic indicators
        to predict diabetes risk, explain key risk factors, and recommend early prevention actions for SDG 3:
        Good Health and Wellbeing.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# KPI ROW
# =========================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Patient Records</div>
            <div class="kpi-value">{total_records:,}</div>
            <div class="kpi-desc">Each record represents one individual health survey response.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Prediction Target</div>
            <div class="kpi-value-teal">Diabetes Status</div>
            <div class="kpi-desc">Classifies whether a person is likely to be no diabetes or prediabetes/diabetes.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Prediabetes / Diabetes</div>
            <div class="kpi-value-pink">{diabetes_percent:.1f}%</div>
            <div class="kpi-desc">Percentage of records labelled as prediabetes or diabetes in the dataset.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Key Risk Signals</div>
            <div class="kpi-value">BMI • BP • Chol</div>
            <div class="kpi-desc">Main health indicators used with lifestyle and demographic variables.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# =========================
# MAIN ROW ONLY
# =========================
left, right = st.columns([1.05, 1.7])

with left:
    st.markdown('<div class="panel-title">🎯 Risk Assessment</div>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability,
        number={
            "suffix": "%",
            "font": {
                "size": 48,
                "color": "#fb3f73" if risk_probability >= 70 else "#f59e0b" if risk_probability >= 40 else "#2dd4bf"
            }
        },
        title={"text": ""},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b98aa"},
            "bar": {"color": "#fb3f73" if risk_probability >= 70 else "#f59e0b" if risk_probability >= 40 else "#2dd4bf"},
            "bgcolor": "#1a2030",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(45, 212, 191, 0.30)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.30)"},
                {"range": [70, 100], "color": "rgba(251, 63, 115, 0.30)"}
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 5},
                "thickness": 0.80,
                "value": risk_probability
            }
        }
    ))

    fig_gauge.update_layout(
        height=340,
        margin=dict(l=15, r=15, t=25, b=5),
        paper_bgcolor="#1a2030",
        plot_bgcolor="#1a2030",
        font=dict(color="#f8fafc")
    )

    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"""
        <div style="text-align:center;">
            <span class="{risk_class}">{risk_level}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown('<div class="panel-title">🧠 Feature Impact Explanation</div>', unsafe_allow_html=True)

    fig_impact = go.Figure()

    fig_impact.add_trace(go.Bar(
        x=impact_df["Impact"],
        y=impact_df["Feature"],
        orientation="h",
        marker=dict(color=impact_colors),
        text=[f"{v:+.2f}" for v in impact_df["Impact"]],
        textposition="outside"
    ))

    fig_impact.add_vline(x=0, line_width=2, line_color="#6b7280")

    fig_impact.update_layout(
        title="Which factors push the risk up or down?",
        height=400,
        margin=dict(l=20, r=40, t=55, b=25),
        paper_bgcolor="#1a2030",
        plot_bgcolor="#1a2030",
        font=dict(color="#f8fafc"),
        xaxis=dict(title="Model impact", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.03)"),
        showlegend=False
    )

    st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})


# =========================
# DETAILED ANALYTICS MOVED LOWER
# =========================
st.markdown(
    """
    <div class="detail-header">
        📌 Detailed Analytics, Findings and Recommended Actions
    </div>
    """,
    unsafe_allow_html=True
)

d1, d2, d3 = st.columns([1, 1, 1])

with d1:
    st.markdown('<div class="panel-title">📊 BMI Pattern</div>', unsafe_allow_html=True)

    fig_bmi = go.Figure()
    fig_bmi.add_trace(go.Bar(
        x=["No Diabetes", "Prediabetes/Diabetes"],
        y=[bmi_summary.loc[0], bmi_summary.loc[1]],
        marker_color=["#2dd4bf", "#fb3f73"],
        text=[f"{bmi_summary.loc[0]:.1f}", f"{bmi_summary.loc[1]:.1f}"],
        textposition="outside"
    ))

    fig_bmi.update_layout(
        height=245,
        margin=dict(l=15, r=15, t=20, b=20),
        paper_bgcolor="#1a2030",
        plot_bgcolor="#1a2030",
        font=dict(color="#f8fafc"),
        yaxis=dict(title="Average BMI", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title="")
    )

    st.plotly_chart(fig_bmi, use_container_width=True, config={"displayModeBar": False})

with d2:
    st.markdown('<div class="panel-title">🩸 Blood Pressure Risk</div>', unsafe_allow_html=True)

    fig_bp = go.Figure()
    fig_bp.add_trace(go.Bar(
        x=["No High BP", "High BP"],
        y=[bp_risk.loc[0], bp_risk.loc[1]],
        marker_color=["#2dd4bf", "#fb3f73"],
        text=[f"{bp_risk.loc[0]:.1f}%", f"{bp_risk.loc[1]:.1f}%"],
        textposition="outside"
    ))

    fig_bp.update_layout(
        height=245,
        margin=dict(l=15, r=15, t=20, b=20),
        paper_bgcolor="#1a2030",
        plot_bgcolor="#1a2030",
        font=dict(color="#f8fafc"),
        yaxis=dict(title="Risk (%)", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title="")
    )

    st.plotly_chart(fig_bp, use_container_width=True, config={"displayModeBar": False})

with d3:
    st.markdown('<div class="panel-title">✅ Finding, Insight & Action</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="finding-box">
        <b>Finding:</b><br>
        High BMI, high blood pressure and older age are strong risk indicators.<br><br>
        <b>Insight:</b><br>
        Early detection allows preventive action before complications.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="action-box">
        <b>Recommended actions:</b><br>
        • Book health screening<br>
        • Monitor blood pressure and cholesterol<br>
        • Improve diet and reduce sugar intake<br>
        • Increase physical activity<br>
        • Track BMI and weight management progress
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer-note">
    Data source: UCI Machine Learning Repository — CDC Diabetes Health Indicators Dataset |
    Data science task: classification | SDG 3: Good Health and Wellbeing
    </div>
    """,
    unsafe_allow_html=True
)
