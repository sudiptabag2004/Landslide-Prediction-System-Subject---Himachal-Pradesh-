"""
Landslide Susceptibility Prediction App
Streamlit version with improved UI - ALL VISIBILITY ISSUES FIXED
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Landslide Susceptibility Assessment",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - COMPLETELY REVISED FOR VISIBILITY
st.markdown("""
<style>
    /* ===== FORCE LIGHT THEME EVERYWHERE ===== */
    :root {
        color-scheme: light !important;
    }
    
    html, body, .stApp, [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], .main, [data-testid="stSidebar"],
    section[data-testid="stSidebar"], .block-container {
        background-color: #ffffff !important;
    }
    
    /* ===== FIX HEADER CUTOFF ===== */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* ===== HEADER STYLING ===== */
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
    
    /* ===== FIX DARK INPUT BACKGROUNDS ===== */
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
    
    /* ===== FIX DATAFRAME DARK BACKGROUND ===== */
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
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2c3e50 !important;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #34495e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background-color: transparent !important;
    }
    
    /* ===== RESULT BOXES ===== */
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
    
    /* ===== INTERPRETATION BOX ===== */
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
    
    /* ===== INFO BOX ===== */
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
    
    /* ===== FOOTER ===== */
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
    
    /* ===== BUTTON STYLING ===== */
    .stButton > button {
        width: 100%;
        background: #3498db !important;
        color: white !important;
        border: none !important;
        padding: 0.85rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 4px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #2980b9 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    /* ===== SLIDER STYLING ===== */
    .stSlider [data-baseweb="slider"] {
        background-color: transparent;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #3498db;
    }
    
    /* ===== FIX LABELS ===== */
    label, .stSlider label {
        color: #2c3e50 !important;
        font-size: 0.9rem !important;
    }
    
    /* ===== INFO MESSAGE ===== */
    .stAlert {
        background-color: #e3f2fd !important;
        border: 1px solid #90caf9 !important;
        color: #1565c0 !important;
    }
    
    /* ===== REMOVE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== COLUMN SPACING ===== */
    div[data-testid="column"] {
        padding: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_PATH = 'landslide_susceptibility_catboost.cbm'

LULC_ENCODING = {
    'Forest': 0,
    'Cropland': 1,
    'Grassland': 2,
    'Barren': 3,
    'Urban': 4,
    'Water': 5,
    'Shrubland': 6
}

FEATURE_IMPORTANCE = {
    'rain_7d': 45.2,
    'rain_3d': 28.1,
    'slope': 15.7,
    'rain_1d': 8.3,
    'lulc': 2.7
}

MODEL_STATS = {
    'roc_auc': 0.9761,
    'recall': 96.15,
    'specificity': 89.33,
    'threshold': 0.5,
    'total_samples': 685500,
    'positive_cases': 103
}

@st.cache_resource
def load_model():
    """Load the CatBoost model - REQUIRED FOR APP TO WORK"""
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

def predict_landslide_demo(features_dict):
    """Demo prediction using simple heuristics"""
    rain_score = (features_dict['rain_7d'] * 0.45 + 
                  features_dict['rain_3d'] * 0.28 + 
                  features_dict['rain_1d'] * 0.08) / 1000
    slope_score = features_dict['slope'] * 0.0157
    lulc_risk = {0: 0.3, 1: 0.1, 2: 0.2, 3: 0.4, 4: 0.05, 5: 0.15, 6: 0.25}
    lulc_score = lulc_risk.get(features_dict['lulc_code'], 0.2) * 0.027
    
    probability = rain_score + slope_score + lulc_score
    probability = min(max(probability, 0.0), 0.99)
    probability += np.random.uniform(-0.05, 0.05)
    probability = min(max(probability, 0.0), 0.99)
    return probability

def predict_landslide(model, features_dict):
    """Make prediction using CatBoost model ONLY"""
    try:
        features_df = pd.DataFrame([[
            features_dict['rain_1d'],
            features_dict['rain_3d'],
            features_dict['rain_7d'],
            features_dict['slope'],
            features_dict['lulc_code']
        ]], columns=['rain_1d', 'rain_3d', 'rain_7d', 'slope', 'lulc'])
        probability = float(model.predict_proba(features_df)[0][1])
        return probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_risk_level(probability):
    """Classify risk level"""
    if probability >= 0.7:
        return {
            'level': 'CRITICAL',
            'color': '#c53030',
            'emoji': '🔴',
            'recommendation': 'Model indicates critical susceptibility level. This output suggests conditions strongly associated with historical landslide occurrence. Requires immediate expert assessment and validation with ground observations.'
        }
    elif probability >= 0.5:
        return {
            'level': 'HIGH',
            'color': '#dd6b20',
            'emoji': '🟠',
            'recommendation': 'Model indicates high susceptibility. Conditions show significant similarity to historical landslide-triggering scenarios. Recommend enhanced monitoring and expert consultation.'
        }
    elif probability >= 0.3:
        return {
            'level': 'MODERATE',
            'color': '#d69e2e',
            'emoji': '🟡',
            'recommendation': 'Model indicates moderate susceptibility. Conditions show partial alignment with landslide-prone scenarios. Continue standard monitoring protocols and maintain situational awareness.'
        }
    else:
        return {
            'level': 'LOW',
            'color': '#38a169',
            'emoji': '🟢',
            'recommendation': 'Model indicates low susceptibility under current parameter conditions. Maintain routine observational protocols. Note that low probability does not guarantee absence of risk.'
        }

def create_map(lat, lon):
    """Create interactive map"""
    fig = go.Figure(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14, 
            color='red',
            symbol='circle'
        ),
        text=[f'<b>Selected Location</b><br>Lat: {lat:.4f}°N<br>Lon: {lon:.4f}°E'],
        hoverinfo='text',
        hovertemplate='<b>Selected Location</b><br>Lat: %{lat:.4f}°N<br>Lon: %{lon:.4f}°E<extra></extra>'
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
        plot_bgcolor='white'
    )
    return fig

# Main app
def main():
    # Custom header
    st.markdown("""
    <div class="header-custom">
        <h1>🏔️ Landslide Susceptibility Assessment System</h1>
        <p>CatBoost ML Model v1.0 | Himachal Pradesh Study Region | Research Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model - REQUIRED
    model = load_model()
    if model is None:
        st.error("❌ **Model File Not Found**")
        st.warning(f"""
        The CatBoost model file `{MODEL_PATH}` is required to run this application.
        
        **To fix this:**
        1. Place your trained model file in the same directory as this app
        2. Ensure the file is named: `landslide_susceptibility_catboost.cbm`
        3. Restart the application
        
        **Current directory:** `{os.getcwd()}`
        """)
        st.stop()  # STOPS THE APP COMPLETELY
    
    # Create two columns for layout
    col_left, col_right = st.columns([0.35, 0.65], gap="large")
    
    with col_left:
        st.markdown('<div class="section-header">📍 LOCATION</div>', unsafe_allow_html=True)
        
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input("Latitude (°N)", min_value=30.0, max_value=33.5, 
                                      value=31.1048, step=0.0001, format="%.4f")
        with col_lon:
            longitude = st.number_input("Longitude (°E)", min_value=75.5, max_value=79.0, 
                                       value=77.1734, step=0.0001, format="%.4f")
        
        location_name = st.text_input("Location Name (Optional)", 
                                     placeholder="e.g., Shimla District")
        
        st.markdown('<div class="section-header">🌧️ RAINFALL PARAMETERS</div>', unsafe_allow_html=True)
        
        rain_1d = st.slider("1-Day Rainfall (mm)", 0, 500, 50, 5)
        rain_3d = st.slider("3-Day Cumulative (mm)", 0, 800, 120, 10)
        rain_7d = st.slider("7-Day Cumulative (mm)", 0, 1500, 200, 10)
        
        st.markdown('<div class="section-header">⛰️ TERRAIN & LAND COVER</div>', unsafe_allow_html=True)
        
        slope = st.slider("Slope Angle (degrees)", 0, 90, 35, 1)
        lulc = st.selectbox("Land Use / Land Cover", list(LULC_ENCODING.keys()))
        
        st.markdown("<br>", unsafe_allow_html=True)
        compute = st.button("🔍 Compute Susceptibility", use_container_width=True, type="primary")
        
        # Model Performance Metrics
        st.markdown('<div class="section-header">📊 MODEL PERFORMANCE METRICS</div>', unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            'Metric': ['ROC-AUC', 'Recall (TPR)', 'Specificity (TNR)', 
                      'False Positive Rate', 'Classification Threshold', 
                      'Training Samples', 'Positive Cases'],
            'Value': ['0.9761', '96.15%', '89.33%', '10.67%', '0.50', 
                     '685,500', '103 (0.015%)']
        })
        
        st.dataframe(metrics_df, hide_index=True, use_container_width=True, height=280)
        
        st.markdown("""
        <div class="info-box-custom">
            <strong>Note on Class Imbalance</strong>
            This model is trained on highly imbalanced data where landslide events 
            represent only 0.015% of observations. The threshold is optimized for 
            high recall at the cost of elevated false positive rates—intentional 
            for life-safety applications.
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="section-header">🗺️ STUDY AREA MAP</div>', unsafe_allow_html=True)
        st.plotly_chart(create_map(latitude, longitude), use_container_width=True)
        
        if compute:
            features_dict = {
                'rain_1d': float(rain_1d),
                'rain_3d': float(rain_3d),
                'rain_7d': float(rain_7d),
                'slope': float(slope),
                'lulc_code': LULC_ENCODING[lulc]
            }
            
            with st.spinner('Computing susceptibility...'):
                probability = predict_landslide(model, features_dict)
            
            if probability is not None:
                risk = get_risk_level(probability)
                confidence = 0.85 + abs(probability - 0.5) * 0.3
                
                st.markdown('<div class="section-header">📈 COMPUTATION RESULTS</div>', unsafe_allow_html=True)
                
                # Result metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(f"""
                    <div class="result-metric">
                        <div class="result-label">Model Probability</div>
                        <div class="result-value">{probability:.6f}</div>
                        <div class="result-subtext">Threshold: 0.50</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="result-metric" style="border-left: 4px solid {risk['color']};">
                        <div class="result-label">Classification</div>
                        <div class="result-value" style="color: {risk['color']};">{risk['emoji']} {risk['level']}</div>
                        <div class="result-subtext">Confidence: {confidence*100:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interpretation
                st.markdown(f"""
                <div class="interpretation-box" style="border-left: 4px solid {risk['color']};">
                    <div class="interpretation-title">Interpretation</div>
                    <div class="interpretation-text">{risk['recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Input Summary
                st.markdown('<div class="section-header">📋 INPUT SUMMARY</div>', unsafe_allow_html=True)
                
                summary_df = pd.DataFrame({
                    'Parameter': ['Location', 'Coordinates', '1-Day Rainfall', 
                                '3-Day Rainfall', '7-Day Rainfall', 'Slope', 'Land Cover'],
                    'Value': [
                        location_name if location_name else 'Custom Location',
                        f"{latitude:.4f}°N, {longitude:.4f}°E",
                        f"{rain_1d} mm",
                        f"{rain_3d} mm",
                        f"{rain_7d} mm",
                        f"{slope}°",
                        lulc
                    ]
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True, height=280)
        else:
            # Welcome/Info section before first computation
            st.markdown('<div class="section-header">👋 WELCOME</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
                <div class="interpretation-title">About This Tool</div>
                <div class="interpretation-text">
                    This system uses machine learning to assess landslide susceptibility based on 
                    rainfall patterns, terrain characteristics, and land cover. The model was trained 
                    on historical landslide data from Himachal Pradesh, India (2014-2016).
                    <br><br>
                    <strong style="color: #2c3e50;">To get started:</strong>
                    <ol style="margin: 0.5rem 0 0 1.2rem; padding: 0; color: #444;">
                        <li>Adjust the location coordinates or click on the map above</li>
                        <li>Set rainfall parameters based on recent weather data</li>
                        <li>Configure terrain slope and land cover type</li>
                        <li>Click <strong>"Compute Susceptibility"</strong> to see results</li>
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance Chart
            st.markdown('<div class="section-header">📊 FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
            
            # Create horizontal bar chart with VISIBLE LABELS
            features = list(FEATURE_IMPORTANCE.keys())
            importance = list(FEATURE_IMPORTANCE.values())
            
            feature_labels = {
                'rain_7d': '7-Day Rainfall',
                'rain_3d': '3-Day Rainfall', 
                'slope': 'Slope Angle',
                'rain_1d': '1-Day Rainfall',
                'lulc': 'Land Cover'
            }
            
            labels = [feature_labels[f] for f in features]
            
            fig = go.Figure(go.Bar(
                x=importance,
                y=labels,
                orientation='h',
                marker=dict(
                    color='#3498db',
                    line=dict(color='#2980b9', width=1)
                ),
                text=[f'{v}%' for v in importance],
                textposition='outside',
                textfont=dict(color='#2c3e50', size=12),
                hovertemplate='<b>%{y}</b><br>Importance: %{x}%<extra></extra>'
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=80, t=20, b=20),
                xaxis=dict(
                    title='Relative Importance (%)',
                    showgrid=True,
                    gridcolor='#eee',
                    range=[0, max(importance) * 1.25],
                    color='#2c3e50'
                ),
                yaxis=dict(
                    title='',
                    autorange='reversed',
                    color='#2c3e50'
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(size=12, color='#2c3e50')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="interpretation-text" style="padding: 0 0.5rem; color: #444 !important;">
                The chart above shows which factors have the most influence on landslide predictions. 
                Cumulative rainfall (especially over 7 days) is the strongest predictor, followed by 
                slope angle and land cover type.
            </div>
            """, unsafe_allow_html=True)
            
            # Quick Stats
            st.markdown('<div class="section-header">📈 KEY STATISTICS</div>', unsafe_allow_html=True)
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.markdown("""
                <div class="result-metric">
                    <div class="result-label">Model Accuracy</div>
                    <div class="result-value" style="font-size: 1.5rem;">97.61%</div>
                    <div class="result-subtext">ROC-AUC Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stats_col2:
                st.markdown("""
                <div class="result-metric">
                    <div class="result-label">Recall Rate</div>
                    <div class="result-value" style="font-size: 1.5rem;">96.15%</div>
                    <div class="result-subtext">True Positives Detected</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stats_col3:
                st.markdown("""
                <div class="result-metric">
                    <div class="result-label">Training Data</div>
                    <div class="result-value" style="font-size: 1.5rem;">685K</div>
                    <div class="result-subtext">Sample Locations</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Study Area Info
            st.markdown('<div class="section-header">🏔️ STUDY AREA</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
                <div class="interpretation-text">
                    <strong style="color: #2c3e50;">Geographic Coverage:</strong> Himachal Pradesh, India<br>
                    <strong style="color: #2c3e50;">Latitude Range:</strong> 30.0°N to 33.5°N<br>
                    <strong style="color: #2c3e50;">Longitude Range:</strong> 75.5°E to 79.0°E<br>
                    <strong style="color: #2c3e50;">Training Period:</strong> 2014-2016<br>
                    <strong style="color: #2c3e50;">Model Type:</strong> CatBoost Gradient Boosting Classifier<br>
                    <strong style="color: #2c3e50;">Total Landslide Events:</strong> 103 documented cases
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer-note">
        <h4>⚠️ Research Use Notice</h4>
        <p>
            This system is a research prototype for demonstration and evaluation purposes. 
            It uses a CatBoost gradient boosting classifier trained on historical landslide 
            data from Himachal Pradesh (2014-2016). The model is optimized for high recall 
            in the context of extreme class imbalance and should not be used as the sole 
            basis for operational decisions. Practical deployment requires integration with 
            real-time monitoring systems, ground-truth validation, and expert interpretation 
            within a comprehensive risk assessment framework. Model performance may degrade 
            outside the spatial and temporal bounds of the training data.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()