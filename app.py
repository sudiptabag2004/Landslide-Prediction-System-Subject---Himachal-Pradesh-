"""
Landslide Susceptibility Prediction App - Enhanced Version
WITH THRESHOLD CONTROL + SHAP EXPLAINABILITY
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import time

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Landslide Susceptibility Assessment",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with subtle animations
st.markdown("""
<style>
    :root {
        color-scheme: light !important;
    }
    
    html, body, .stApp, [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], .main, [data-testid="stSidebar"],
    section[data-testid="stSidebar"], .block-container {
        background-color: #ffffff !important;
    }
    
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    .header-custom {
        background: #2c3e50;
        color: white;
        padding: 1.5rem 2rem;
        margin: -2rem -1rem 2rem -1rem;
        border-bottom: 3px solid #34495e;
    }
    
    .header-custom h1 {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        color: white !important;
    }
    
    .header-custom p {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.95) !important;
        margin: 0;
    }
    
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #ccc !important;
    }
    
    .stDataFrame, 
    [data-testid="stDataFrame"],
    .stDataFrame > div,
    .dataframe {
        background-color: #ffffff !important;
    }
    
    .stDataFrame td,
    .stDataFrame th,
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    .section-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2c3e50 !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #34495e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background-color: transparent !important;
    }
    
    .result-metric {
        background: #ffffff;
        border: 1px solid #ddd;
        padding: 1.5rem;
        text-align: center;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .result-label {
        font-size: 0.75rem;
        color: #666 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50 !important;
        margin: 0.5rem 0;
    }
    
    .result-subtext {
        font-size: 0.85rem;
        color: #666 !important;
    }
    
    .interpretation-box {
        background: #ffffff;
        border: 1px solid #ddd;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .interpretation-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #2c3e50 !important;
    }
    
    .interpretation-text {
        font-size: 0.9rem;
        line-height: 1.7;
        color: #444 !important;
    }
    
    .info-box-custom {
        background: #fffbf0;
        border: 1px solid #ffc107;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        font-size: 0.85rem;
        line-height: 1.7;
        margin-top: 1rem;
        border-radius: 4px;
    }
    
    .info-box-custom,
    .info-box-custom * {
        color: #333 !important;
    }
    
    .info-box-custom strong {
        display: block;
        margin-bottom: 0.5rem;
        color: #000 !important;
        font-size: 0.9rem;
    }
    
    .alert-box {
        background: #fee;
        border: 2px solid #c53030;
        border-left: 6px solid #c53030;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-weight: 600;
        text-align: center;
    }
    
    .no-alert-box {
        background: #e8f5e9;
        border: 2px solid #38a169;
        border-left: 6px solid #38a169;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-weight: 600;
        text-align: center;
    }
    
    .alert-box-text {
        font-size: 1.1rem;
        color: #c53030 !important;
        margin: 0;
    }
    
    .no-alert-box-text {
        font-size: 1.1rem;
        color: #38a169 !important;
        margin: 0;
    }
    
    .footer-note {
        background: #ecf0f1;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 4px solid #95a5a6;
        border-radius: 4px;
    }
    
    .footer-note h4 {
        margin: 0 0 0.75rem 0;
        color: #2c3e50 !important;
        font-size: 1rem;
    }
    
    .footer-note p {
        font-size: 0.85rem;
        line-height: 1.7;
        color: #444 !important;
        margin: 0;
    }
    
    .stButton > button {
        width: 100%;
        background: #3498db !important;
        color: white !important;
        border: none !important;
        padding: 0.85rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 4px;
        transition: all 0.2s ease-out;
    }
    
    .stButton > button:hover {
        background: #2980b9 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }

    .stButton > button:active {
        transform: scale(0.98);
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: transparent;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #3498db;
    }
    
    label, .stSlider label {
        color: #2c3e50 !important;
        font-size: 0.9rem !important;
    }
    
    .stAlert {
        background-color: #e3f2fd !important;
        border: 1px solid #90caf9 !important;
        color: #1565c0 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div[data-testid="column"] {
        padding: 0 0.5rem;
    }

    /* Fade-in animation */
    .fade-in {
        opacity: 0;
        animation: fadeIn 0.4s ease-in-out forwards;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* One-time pulse for alert box */
    .pulse-once {
        animation: pulseOnce 0.6s ease-out;
    }

    @keyframes pulseOnce {
        0% { box-shadow: 0 0 0 0 rgba(197,48,48,0.6); }
        100% { box-shadow: 0 0 12px rgba(197,48,48,0); }
    }

    /* Map caption */
    .map-caption {
        font-size: 0.8rem;
        color: #555;
        opacity: 0.95;
        transition: opacity 0.3s ease-in-out;
    }

    .map-caption:hover {
        opacity: 1.0;
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_PATH = 'landslide_susceptibility_catboost.cbm'
HIST_CSV_PATH = 'landslides_hp.csv'

FEATURE_ORDER = ['rain_1d', 'rain_3d', 'rain_7d', 'slope', 'lulc']

LULC_ENCODING = {
    'Forest': 0,
    'Cropland': 1,
    'Grassland': 2,
    'Barren': 3,
    'Urban': 4,
    'Water': 5,
    'Shrubland': 6
}

THRESHOLD_INFO = {
    0.1: {"recall": 99.03, "alert_rate": 28.45, "specificity": 71.55},
    0.2: {"recall": 98.06, "alert_rate": 18.92, "specificity": 81.08},
    0.3: {"recall": 97.09, "alert_rate": 14.33, "specificity": 85.67},
    0.4: {"recall": 96.60, "alert_rate": 12.18, "specificity": 87.82},
    0.5: {"recall": 96.15, "alert_rate": 10.67, "specificity": 89.33},
    0.6: {"recall": 89.32, "alert_rate": 7.24, "specificity": 92.76},
    0.7: {"recall": 77.67, "alert_rate": 4.21, "specificity": 95.79},
    0.8: {"recall": 58.25, "alert_rate": 1.89, "specificity": 98.11}
}

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            return None
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_historical_landslides():
    if not os.path.exists(HIST_CSV_PATH):
        return None
    try:
        df = pd.read_csv(HIST_CSV_PATH)
        required_cols = {'event_date', 'latitude', 'longitude'}
        if not required_cols.issubset(df.columns):
            return None
        df = df.dropna(subset=['latitude', 'longitude'])
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df.dropna(subset=['event_date'])
        return df
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2_array, lon2_array):
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_historical_context(lat, lon, buffer_km=20.0):
    """Return summary stats plus dataframe of nearby events."""
    hist_df = load_historical_landslides()
    if hist_df is None or hist_df.empty:
        return None, None
    dists_km = haversine_km(lat, lon, hist_df['latitude'].values, hist_df['longitude'].values)
    hist_df = hist_df.copy()
    hist_df['distance_km'] = dists_km
    nearby = hist_df[hist_df['distance_km'] <= buffer_km]
    if nearby.empty:
        ctx = {'count': 0, 'nearest_km': None, 'recent_year': None}
        return ctx, nearby
    ctx = {
        'count': int(len(nearby)),
        'nearest_km': float(nearby['distance_km'].min()),
        'recent_year': int(nearby['event_date'].max().year)
    }
    return ctx, nearby

def predict_landslide(model, features_dict):
    try:
        row = [
            features_dict[f] if f != 'lulc' else features_dict['lulc_code']
            for f in FEATURE_ORDER
        ]
        features_df = pd.DataFrame([row], columns=FEATURE_ORDER)
        probability = float(model.predict_proba(features_df)[0][1])
        return probability, features_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def get_risk_level_by_threshold(probability, threshold):
    eps = 1e-6
    if probability >= threshold:
        distance = (probability - threshold) / max((1.0 - threshold), eps)
        if distance >= 0.7:
            return {
                'level': 'VERY HIGH',
                'color': '#9b1c1c',
                'emoji': '🔴',
                'alert': True,
                'recommendation': (
                    f'Predicted probability P(landslide) = {probability:.4f} significantly exceeds threshold ({threshold:.2f}). '
                    'Model indicates very high susceptibility. Immediate expert assessment and '
                    'validation with ground observations are recommended.'
                )
            }
        elif distance >= 0.4:
            return {
                'level': 'HIGH',
                'color': '#c53030',
                'emoji': '🟠',
                'alert': True,
                'recommendation': (
                    f'Predicted probability P(landslide) = {probability:.4f} exceeds threshold ({threshold:.2f}). '
                    'Model indicates high susceptibility. Enhanced monitoring and risk mitigation '
                    'measures are recommended.'
                )
            }
        else:
            return {
                'level': 'MODERATE',
                'color': '#dd6b20',
                'emoji': '🟡',
                'alert': True,
                'recommendation': (
                    f'Predicted probability P(landslide) = {probability:.4f} exceeds threshold ({threshold:.2f}). '
                    'Model indicates moderate susceptibility. Continue monitoring with heightened awareness.'
                )
            }
    else:
        return {
            'level': 'LOW',
            'color': '#38a169',
            'emoji': '🟢',
            'alert': False,
            'recommendation': (
                f'Predicted probability P(landslide) = {probability:.4f} is below threshold ({threshold:.2f}). '
                'Model indicates low susceptibility under current conditions. Maintain routine monitoring protocols.'
            )
        }

@st.cache_resource
def get_shap_explainer(model):
    import shap
    return shap.TreeExplainer(model)

def get_shap_explanation(model, features_df):
    try:
        import shap
        explainer = get_shap_explainer(model)
        shap_values = explainer.shap_values(features_df)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[-1])
        else:
            base_value = float(base_value)
        if isinstance(shap_values, list):
            shap_vals = shap_values[-1][0]
        else:
            shap_vals = shap_values[0]
        feature_names = features_df.columns.tolist()
        feature_values = features_df.iloc[0].tolist()
        contributions = {}
        for fname, fval, shap_val in zip(feature_names, feature_values, shap_vals):
            contributions[fname] = {
                'value': fval,
                'contribution': float(shap_val),
                'abs_contribution': abs(float(shap_val))
            }
        sorted_features = sorted(
            contributions.items(), 
            key=lambda x: x[1]['abs_contribution'], 
            reverse=True
        )
        return {
            'base_value': base_value,
            'contributions': dict(sorted_features),
            'prediction_explained': True
        }
    except ImportError:
        feature_names = features_df.columns.tolist()
        feature_values = features_df.iloc[0].tolist()
        importance_map = {
            'rain_7d': 0.452,
            'rain_3d': 0.281,
            'slope': 0.157,
            'rain_1d': 0.083,
            'lulc': 0.027
        }
        contributions = {}
        for fname, fval in zip(feature_names, feature_values):
            if fname.startswith('rain'):
                norm_val = fval / 1000.0
            elif fname == 'slope':
                norm_val = fval / 90.0
            else:
                norm_val = fval / 10.0
            contrib_val = norm_val * importance_map.get(fname, 0.1)
            contributions[fname] = {
                'value': fval,
                'contribution': contrib_val,
                'abs_contribution': abs(contrib_val)
            }
        sorted_features = sorted(
            contributions.items(), 
            key=lambda x: x[1]['abs_contribution'], 
            reverse=True
        )
        return {
            'base_value': 0.5,
            'contributions': dict(sorted_features),
            'prediction_explained': False
        }
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")
        return None

def create_shap_plot(explanation):
    """Create visualization of feature contributions with high-contrast labels."""
    if explanation is None:
        return None

    feature_labels = {
        'rain_7d': '7-Day Rainfall',
        'rain_3d': '3-Day Rainfall',
        'slope': 'Slope Angle',
        'rain_1d': '1-Day Rainfall',
        'lulc': 'Land Cover'
    }

    features = []
    contributions = []
    colors = []
    for fname, data in explanation['contributions'].items():
        features.append(feature_labels.get(fname, fname))
        contrib = data['contribution']
        contributions.append(contrib)
        colors.append('#e74c3c' if contrib > 0 else '#27ae60')

    fig = go.Figure(go.Bar(
        y=features[::-1],
        x=contributions[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='#2c3e50', width=1)
        ),
        text=[f'{v:+.4f}' for v in contributions[::-1]],
        textposition='outside',
        textfont=dict(color='#2c3e50', size=11),
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=80, t=20, b=40),
        xaxis=dict(
            title='Contribution to Prediction (SHAP, log-odds)',
            showgrid=True,
            gridcolor='#eeeeee',
            zeroline=True,
            zerolinecolor='#2c3e50',
            zerolinewidth=1.5,
            color='#2c3e50',
            tickfont=dict(color='#2c3e50')
        ),
        yaxis=dict(
            title='',
            color='#2c3e50',
            tickfont=dict(color='#2c3e50')
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=12, color='#2c3e50')
    )

    return fig

def create_map(lat, lon, nearby_events=None):
    """Map with selected location + optional nearby historical landslides."""
    fig = go.Figure()

    # Selected location marker
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=16,
            color='red',
            symbol='circle'
        ),
        text=[f'Selected Location<br>Lat: {lat:.4f}°N<br>Lon: {lon:.4f}°E'],
        name='Selected location',
        hoverinfo='text'
    ))

    # Nearby historical events as small blue points
    if nearby_events is not None and not nearby_events.empty:
        fig.add_trace(go.Scattermapbox(
            lat=nearby_events['latitude'],
            lon=nearby_events['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color='blue',
                opacity=0.8
            ),
            text=[
                f"Historical landslide<br>Date: {d.date()}<br>Dist: {dist:.2f} km"
                for d, dist in zip(nearby_events['event_date'], nearby_events['distance_km'])
            ],
            name='Historical landslides (≤20 km)',
            hoverinfo='text'
        ))

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=lat, lon=lon),
            zoom=8
        ),
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.01,
            xanchor='left',
            x=0.01,
        ),
        legend_font_color="#2c3e50",
        legend_font_size=11
    )
    return fig

def main():
    st.markdown("""
    <div class="header-custom">
        <h1>Landslide Susceptibility Assessment System</h1>
        <p>CatBoost ML Model v1.0 | Himachal Pradesh Study Region | Research Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        st.error("Model file not found.")
        st.stop()
    
    col_left, col_right = st.columns([0.35, 0.65], gap="large")
    
    with col_left:
        st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input(
                "Latitude (°N)", min_value=30.0, max_value=33.5, 
                value=31.1048, step=0.0001, format="%.4f"
            )
        with col_lon:
            longitude = st.number_input(
                "Longitude (°E)", min_value=75.5, max_value=79.0, 
                value=77.1734, step=0.0001, format="%.4f"
            )
        
        st.markdown('<div class="section-header">Rainfall parameters</div>', unsafe_allow_html=True)
        rain_1d = st.slider("1-day rainfall (mm)", 0, 500, 50, 5)
        rain_3d = st.slider("3-day cumulative (mm)", 0, 800, 120, 10)
        rain_7d = st.slider("7-day cumulative (mm)", 0, 1500, 200, 10)
        
        st.markdown('<div class="section-header">Terrain & land cover</div>', unsafe_allow_html=True)
        slope = st.slider("Slope angle (degrees)", 0, 90, 35, 1)
        lulc = st.selectbox("Land use / land cover", list(LULC_ENCODING.keys()))
        
        st.markdown('<div class="section-header">Decision threshold</div>', unsafe_allow_html=True)
        threshold = st.slider(
            "Classification threshold", 
            min_value=0.1, 
            max_value=0.8, 
            value=0.5, 
            step=0.1,
            help="Lower threshold increases recall at the cost of more alerts."
        )
        thresh_info = THRESHOLD_INFO.get(threshold, THRESHOLD_INFO[0.5])
        st.markdown(f"""
        <div class="info-box-custom fade-in">
            <strong>Threshold performance (@ {threshold:.1f})</strong>
            • Recall (sensitivity): {thresh_info['recall']:.2f}%<br>
            • Alert rate (% classified positive): {thresh_info['alert_rate']:.2f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        compute = st.button("Compute susceptibility", use_container_width=True, type="primary")
    
    with col_right:
        st.markdown('<div class="section-header">Study area map</div>', unsafe_allow_html=True)
        map_placeholder = st.empty()
        map_placeholder.plotly_chart(
            create_map(latitude, longitude, nearby_events=None),
            use_container_width=True
        )
        st.markdown("""
        <div class="map-caption">
            Map visualization is for spatial context only and does not represent model grid resolution.
        </div>
        """, unsafe_allow_html=True)
        
        if compute:
            features_dict = {
                'rain_1d': float(rain_1d),
                'rain_3d': float(rain_3d),
                'rain_7d': float(rain_7d),
                'slope': float(slope),
                'lulc_code': LULC_ENCODING[lulc]
            }
            with st.spinner('Computing susceptibility and explanation...'):
                probability, features_df = predict_landslide(model, features_dict)
                explanation = None
                if probability is not None:
                    explanation = get_shap_explanation(model, features_df)
            
            # Alert / no-alert
            alert_placeholder = st.empty()
            if probability is not None:
                risk = get_risk_level_by_threshold(probability, threshold)
                if risk['alert']:
                    alert_html = """
                    <div class="alert-box pulse-once fade-in">
                        <p class="alert-box-text">Alert: susceptibility exceeds threshold</p>
                    </div>
                    """
                else:
                    alert_html = """
                    <div class="no-alert-box fade-in">
                        <p class="no-alert-box-text">No alert: below threshold</p>
                    </div>
                    """
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
                time.sleep(0.15)
            
            # Probability + classification
            metrics_placeholder = st.empty()
            if probability is not None:
                with metrics_placeholder.container():
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.markdown(f"""
                        <div class="result-metric fade-in">
                            <div class="result-label">Predicted probability P(landslide)</div>
                            <div class="result-value">{probability:.6f}</div>
                            <div class="result-subtext">Current threshold: {threshold:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with metric_col2:
                        st.markdown(f"""
                        <div class="result-metric fade-in" style="border-left: 4px solid {risk['color']};">
                            <div class="result-label">Classification</div>
                            <div class="result-value" style="color: {risk['color']};">
                                {risk['emoji']} {risk['level']}
                            </div>
                            <div class="result-subtext">Threshold-based risk category</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="interpretation-box fade-in" style="border-left: 4px solid {risk['color']};">
                        <div class="interpretation-title">Interpretation</div>
                        <div class="interpretation-text">{risk['recommendation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                time.sleep(0.15)
            
            st.markdown('<hr style="border:none; border-top:1px solid #eee; margin:1.2rem 0;">', unsafe_allow_html=True)
            
            # Historical context + update same map with historical points
            hist_placeholder = st.empty()
            with hist_placeholder.container():
                st.markdown('<div class="section-header">Historical landslides</div>', unsafe_allow_html=True)
                hist_ctx, nearby_events = compute_historical_context(latitude, longitude, buffer_km=20.0)
                if hist_ctx is None:
                    st.markdown("""
                    <div class="interpretation-box fade-in">
                        <div class="interpretation-title">Historical Landslide Context (within 20 km)</div>
                        <div class="interpretation-text">
                            Historical landslide catalog could not be loaded from <code>landslides_hp.csv</code>.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if hist_ctx['count'] == 0:
                        st.markdown("""
                        <div class="interpretation-box fade-in">
                            <div class="interpretation-title">Historical Landslide Context (within 20 km)</div>
                            <div class="interpretation-text">
                                No recorded historical landslides within 20 km of this location.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="interpretation-box fade-in">
                            <div class="interpretation-title">Historical Landslide Context (within 20 km)</div>
                            <div class="interpretation-text">
                                • Total past landslides within 20 km: <strong>{hist_ctx['count']}</strong><br>
                                • Nearest historical landslide: <strong>{hist_ctx['nearest_km']:.2f} km</strong><br>
                                • Most recent event year: <strong>{hist_ctx['recent_year']}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("""
                <div class="interpretation-text fade-in" style="font-size:0.8rem; color:#555; margin-top:0.5rem;">
                    Historical events are shown for situational awareness only and are independent of model predictions.
                </div>
                """, unsafe_allow_html=True)

            # Update the same map placeholder with historical events overlay
            map_placeholder.plotly_chart(
                create_map(latitude, longitude, nearby_events=nearby_events),
                use_container_width=True
            )

            st.markdown('<hr style="border:none; border-top:1px solid #eee; margin:1.2rem 0;">', unsafe_allow_html=True)
            
            # SHAP explanation
            shap_placeholder = st.empty()
            if probability is not None and explanation is not None:
                with shap_placeholder.container():
                    st.markdown('<div class="section-header">Prediction explanation</div>', unsafe_allow_html=True)
                    shap_fig = create_shap_plot(explanation)
                    if shap_fig is not None:
                        st.plotly_chart(shap_fig, use_container_width=True)
                    top_features = list(explanation['contributions'].items())[:3]
                    feature_labels = {
                        'rain_7d': '7-Day Rainfall',
                        'rain_3d': '3-Day Rainfall',
                        'slope': 'Slope Angle',
                        'rain_1d': '1-Day Rainfall',
                        'lulc': 'Land Cover'
                    }
                    explanation_text = (
                        "<div class='interpretation-box fade-in'>"
                        "<div class='interpretation-title'>Key contributors</div>"
                        "<div class='interpretation-text'>"
                    )
                    for fname, data in top_features:
                        direction = "increased" if data['contribution'] > 0 else "decreased"
                        arrow = "↑" if data['contribution'] > 0 else "↓"
                        explanation_text += (
                            f"<strong style='color: #2c3e50;'>{arrow} {feature_labels.get(fname, fname)}</strong>: "
                            f"Value = {data['value']:.2f}, {direction} susceptibility by "
                            f"{abs(data['contribution']):.4f}<br>"
                        )
                    if explanation['prediction_explained']:
                        explanation_text += (
                            "<br>This explanation is based on SHAP values from the CatBoost model, "
                            "which attribute the prediction to each feature in a locally consistent way."
                        )
                    else:
                        explanation_text += (
                            "<br>This explanation uses an importance-based approximation because "
                            "the SHAP library was not available in the runtime environment."
                        )
                    explanation_text += "</div></div>"
                    st.markdown(explanation_text, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">Welcome</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="interpretation-box fade-in">
                <div class="interpretation-title">About this dashboard</div>
                <div class="interpretation-text">
                    This web-based system is an inference-only research dashboard that applies a
                    pre-trained CatBoost susceptibility model to user-defined environmental scenarios.
                    It does not ingest real-time data nor issue public warnings, and is intended solely
                    for decision-support, sensitivity analysis, and scientific demonstration.
                    <br><br>
                    <strong style="color:#2c3e50;">How to use</strong>
                    <ol style="margin:0.5rem 0 0 1.2rem; padding:0; color:#444;">
                        <li>Select a location within the Himachal Pradesh study area.</li>
                        <li>Set rainfall parameters based on hypothetical or historical conditions.</li>
                        <li>Configure slope and land cover.</li>
                        <li>Adjust the decision threshold to explore recall vs. alert tradeoffs.</li>
                        <li>Click <strong>“Compute susceptibility”</strong> to view risk category and explanation.</li>
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer-note">
        <h4>Research use notice</h4>
        <p>
            The web-based system is an inference-only research dashboard that applies a pre-trained CatBoost
            susceptibility model to user-defined environmental scenarios. It does not ingest real-time data
            nor issue public warnings, and is intended solely for decision-support, sensitivity analysis,
            and scientific demonstration. Any operational use would require additional validation, real-time
            data integration, and expert oversight.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
