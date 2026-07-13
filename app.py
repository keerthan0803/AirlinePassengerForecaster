import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------
# Design tokens — "Executive Brief" identity
# Clean white background, navy for headings,
# blue for forecasts, slate-teal for historical data,
# crisp sans-serif type for a professional dashboard feel.
# ------------------------------------------------

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');

    :root {
        --bg-void: #F7F9FC;
        --panel: #FFFFFF;
        --panel-alt: #F3F5F9;
        --border-soft: #E2E7F0;
        --navy: #1E2A4A;
        --blue: #2F5FDE;
        --teal: #0E8388;
        --text-primary: #1B2436;
        --text-muted: #5C6A85;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(47,95,222,0.03), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(14,131,136,0.03), transparent 35%),
            var(--bg-void);
        color: var(--text-primary);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 2px solid #1B2436 !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    section[data-testid="stSidebar"] .stSlider label {
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.75rem;
        color: var(--text-muted) !important;
    }
    section[data-testid="stSidebar"] h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--navy) !important;
        border-bottom: 2px dashed #1B2436;
        padding-bottom: 0.6rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stAlert"] {
        background: var(--panel-alt) !important;
        border: 2px solid #1B2436 !important;
        border-left: 6px solid var(--blue) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        box-shadow: 4px 4px 0px 0px #1B2436 !important;
    }

    /* ---------- Hero / masthead ---------- */
    .board-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding: 1.5rem 1.8rem;
        margin-bottom: 0.4rem;
        background: var(--panel) !important;
        border: 2px solid #1B2436 !important;
        border-radius: 12px !important;
        box-shadow: 4px 4px 0px 0px #1B2436 !important;
    }
    .board-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--teal);
        margin-bottom: 0.35rem;
    }
    .board-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        line-height: 1.1;
        color: var(--navy);
        margin: 0;
    }
    .board-caption {
        font-family: 'Inter', sans-serif;
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }
    .board-route {
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
        color: var(--text-muted);
        font-size: 0.75rem;
        letter-spacing: 0.1em;
    }
    .board-route span { color: var(--blue); font-weight: 700; }

    /* dashed flight-path divider */
    .flight-path {
        height: 18px;
        margin: 0 0 1.3rem 0;
        background-image: radial-gradient(circle, #1B2436 1.8px, transparent 1.8px);
        background-size: 14px 14px;
        background-position: center;
        opacity: 0.9;
    }

    /* ---------- Section headers ---------- */
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--navy) !important;
        letter-spacing: 0.01em;
    }
    .stMarkdown h2 {
        border-left: 3px solid var(--blue);
        padding-left: 0.6rem;
    }

    /* ---------- Metrics as report tiles ---------- */
    div[data-testid="stMetric"] {
        background: var(--panel) !important;
        border: 2px solid #1B2436 !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 4px 4px 0px 0px #1B2436 !important;
    }
    div[data-testid="stMetric"] label {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.7rem !important;
        color: var(--text-muted) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--blue) !important;
        font-size: 1.7rem !important;
    }

    /* ---------- Tabs as boarding gates ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #1B2436 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--panel-alt) !important;
        border: 2px solid #1B2436 !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.6rem 1.1rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
        color: var(--text-muted) !important;
    }
    .stTabs [data-baseweb="tab"] * {
        color: var(--text-muted) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--blue) !important;
        border-color: var(--blue) !important;
    }
    .stTabs [data-baseweb="tab"]:hover * {
        color: var(--blue) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        background: var(--blue) !important;
        border-color: #1B2436 !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button:first-child {
        background: var(--blue) !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        border: 2px solid #1B2436 !important;
        border-radius: 8px !important;
        height: 3.2em !important;
        box-shadow: 4px 4px 0px 0px #1B2436 !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
        cursor: pointer !important;
    }
    div.stButton > button:first-child:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px 0px #1B2436 !important;
    }
    div.stButton > button:first-child:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px 0px #1B2436 !important;
    }

    /* ---------- Download button ---------- */
    div[data-testid="stDownloadButton"] > button {
        background: var(--panel) !important;
        color: var(--teal) !important;
        border: 2px solid #1B2436 !important;
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        width: 100% !important;
        box-shadow: 4px 4px 0px 0px #1B2436 !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: var(--panel-alt) !important;
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px 0px #1B2436 !important;
    }

    /* ---------- Dataframes ---------- */
    div[data-testid="stDataFrame"] {
        border: 2px solid #1B2436 !important;
        border-radius: 8px !important;
        box-shadow: 4px 4px 0px 0px #1B2436 !important;
        overflow: hidden !important;
    }

    /* ---------- Divider ---------- */
    hr { border-color: #1B2436 !important; border-width: 2px !important; }

    /* ---------- Success banner ---------- */
    div[data-testid="stAlert"][kind="success"] {
        border-left: 6px solid var(--teal) !important;
    }

    /* ---------- Header bar transparency ---------- */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] * {
        color: var(--text-primary) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Sidebar & Logic
# ------------------------------------------------

with st.sidebar:
    st.image("assets/201623.png", width=100)
    st.title("Flight Deck Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.info("Adjust the slider to change the prediction window for the RNN model.")

# ------------------------------------------------
# Data & Masthead
# ------------------------------------------------

loader = DataLoader("data/airline_passengers.csv")
df = loader.load_data()

st.markdown("""
    <div class="board-header">
        <div>
            <div class="board-eyebrow">Terminal Analytics · Live Model</div>
            <p class="board-title">✈ Airline Passenger Forecaster</p>
            <p class="board-caption">Predicting global travel trends using Recurrent Neural Networks (RNN)</p>
        </div>
        <div class="board-route">GATE <span>PAX-01</span><br/>STATUS <span>ON TIME</span></div>
    </div>
    <div class="flight-path"></div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Metrics & Overview Tabs
# ------------------------------------------------

tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])

with tab1:
    st.subheader("Model Accuracy Metrics")
    mae, mse, rmse = Evaluator().evaluate()

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", delta_color="inverse")
    m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}", delta_color="inverse")
    m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", delta_color="inverse")

with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Raw Data")
        st.dataframe(df, height=350)

    with col_b:
        st.subheader("Historical Trend")
        fig = px.line(df, x=df.index, y="Passengers",
                      template="plotly_white",
                      color_discrete_sequence=['#0E8388'])
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#1B2436"),
        )
        fig.update_xaxes(gridcolor="#E2E7F0")
        fig.update_yaxes(gridcolor="#E2E7F0")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecasting Section
# ------------------------------------------------

st.markdown("---")
st.header("🔮 Generate Future Forecast")

if st.button("Run RNN Model"):
    with st.spinner("Analyzing temporal patterns..."):
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)

        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })

    st.success(f"Successfully generated forecast for {future_months} months!")

    # Layout for Results
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    with res_col2:
        st.subheader("Combined Projection")

        # Create a combined chart with Plotly
        fig_combined = go.Figure()

        # Historical Data
        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df["Passengers"],
            name="Historical", line=dict(color="#6B7688", width=2)
        ))

        # Forecasted Data
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="Forecast", line=dict(color="#2F5FDE", width=3, dash='dot')
        ))

        fig_combined.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#1B2436"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_combined.update_xaxes(gridcolor="#E2E7F0")
        fig_combined.update_yaxes(gridcolor="#E2E7F0")
        st.plotly_chart(fig_combined, use_container_width=True)