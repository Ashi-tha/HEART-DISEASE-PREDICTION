"""
Heart Disease Prediction System - Streamlit App
A beautiful, interactive web application for heart disease risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #d32f2f;
        font-weight: 700;
    }
    h2, h3 {
        color: #1976d2;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run train_model.py first.")
        st.stop()

# Feature information
FEATURE_INFO = {
    'age': {
        'name': 'Age',
        'min': 20, 'max': 100, 'default': 50,
        'help': 'Patient age in years'
    },
    'sex': {
        'name': 'Sex',
        'options': ['Female', 'Male'],
        'help': 'Biological sex of the patient'
    },
    'cp': {
        'name': 'Chest Pain Type',
        'options': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
        'help': 'Type of chest pain experienced'
    },
    'trestbps': {
        'name': 'Resting Blood Pressure',
        'min': 80, 'max': 200, 'default': 120,
        'help': 'Blood pressure at rest (mm Hg). Normal: ~120'
    },
    'chol': {
        'name': 'Cholesterol',
        'min': 100, 'max': 600, 'default': 200,
        'help': 'Serum cholesterol (mg/dl). Normal: <200'
    },
    'fbs': {
        'name': 'Fasting Blood Sugar > 120 mg/dl',
        'options': ['No', 'Yes'],
        'help': 'Whether fasting blood sugar exceeds 120 mg/dl'
    },
    'restecg': {
        'name': 'Resting ECG',
        'options': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
        'help': 'Resting electrocardiographic results'
    },
    'thalach': {
        'name': 'Maximum Heart Rate',
        'min': 60, 'max': 220, 'default': 150,
        'help': 'Maximum heart rate achieved (bpm). Age-predicted max: ~220 - age'
    },
    'exang': {
        'name': 'Exercise Induced Angina',
        'options': ['No', 'Yes'],
        'help': 'Whether exercise causes chest pain'
    },
    'oldpeak': {
        'name': 'ST Depression',
        'min': 0.0, 'max': 7.0, 'default': 1.0,
        'help': 'ST depression induced by exercise relative to rest (mm)'
    },
    'slope': {
        'name': 'Slope of Peak Exercise ST',
        'options': ['Upsloping', 'Flat', 'Downsloping'],
        'help': 'Slope of the peak exercise ST segment'
    },
    'ca': {
        'name': 'Number of Major Vessels',
        'options': ['0', '1', '2', '3'],
        'help': 'Number of major vessels colored by fluoroscopy'
    },
    'thal': {
        'name': 'Thalassemia',
        'options': ['Normal', 'Fixed Defect', 'Reversible Defect'],
        'help': 'Blood disorder status'
    }
}

def create_gauge_chart(value, title):
    """Create a gauge chart for risk visualization"""
    
    if value < 30:
        color = "green"
    elif value < 60:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def main():
    """Main application"""
    
    # Title and description
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 0.5rem; color: white; margin-bottom: 2rem;'>
        <h3 style='color: white; margin: 0;'>AI-Powered Cardiovascular Risk Assessment</h3>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
            This system uses machine learning to predict heart disease risk based on clinical parameters.
            <strong>For research and educational purposes only.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(f"""
        **Model:** Support Vector Machine (SVM)
        
        **Performance:**
        - Accuracy: 83.61%
        - F1 Score: 86.11%
        - Recall: 93.94%
        
        **Features:** 13 clinical parameters
        
        **Dataset:** 302 patient records
        """)
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        st.metric("Model Type", "SVM")
        st.metric("Training Samples", "302")
        st.metric("Accuracy", "83.61%")
        
        st.markdown("---")
        st.warning("⚠️ **Medical Disclaimer**: This tool is for research purposes only. Always consult healthcare professionals for medical advice.")
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🏥 Patient Information")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🩺 Clinical Measurements", "❤️ Cardiac Assessment"])
        
        # Dictionary to store inputs
        inputs = {}
        
        with tab1:
            st.subheader("Patient Demographics")
            c1, c2 = st.columns(2)
            
            with c1:
                inputs['age'] = st.slider(
                    FEATURE_INFO['age']['name'],
                    FEATURE_INFO['age']['min'],
                    FEATURE_INFO['age']['max'],
                    FEATURE_INFO['age']['default'],
                    help=FEATURE_INFO['age']['help']
                )
            
            with c2:
                sex_val = st.selectbox(
                    FEATURE_INFO['sex']['name'],
                    FEATURE_INFO['sex']['options'],
                    help=FEATURE_INFO['sex']['help']
                )
                inputs['sex'] = 1 if sex_val == 'Male' else 0
        
        with tab2:
            st.subheader("Clinical Measurements")
            
            c1, c2 = st.columns(2)
            
            with c1:
                inputs['trestbps'] = st.slider(
                    FEATURE_INFO['trestbps']['name'],
                    FEATURE_INFO['trestbps']['min'],
                    FEATURE_INFO['trestbps']['max'],
                    FEATURE_INFO['trestbps']['default'],
                    help=FEATURE_INFO['trestbps']['help']
                )
                
                inputs['chol'] = st.slider(
                    FEATURE_INFO['chol']['name'],
                    FEATURE_INFO['chol']['min'],
                    FEATURE_INFO['chol']['max'],
                    FEATURE_INFO['chol']['default'],
                    help=FEATURE_INFO['chol']['help']
                )
            
            with c2:
                fbs_val = st.selectbox(
                    FEATURE_INFO['fbs']['name'],
                    FEATURE_INFO['fbs']['options'],
                    help=FEATURE_INFO['fbs']['help']
                )
                inputs['fbs'] = 1 if fbs_val == 'Yes' else 0
                
                inputs['thalach'] = st.slider(
                    FEATURE_INFO['thalach']['name'],
                    FEATURE_INFO['thalach']['min'],
                    FEATURE_INFO['thalach']['max'],
                    FEATURE_INFO['thalach']['default'],
                    help=FEATURE_INFO['thalach']['help']
                )
        
        with tab3:
            st.subheader("Cardiac Assessment")
            
            c1, c2 = st.columns(2)
            
            with c1:
                cp_val = st.selectbox(
                    FEATURE_INFO['cp']['name'],
                    FEATURE_INFO['cp']['options'],
                    help=FEATURE_INFO['cp']['help']
                )
                inputs['cp'] = FEATURE_INFO['cp']['options'].index(cp_val)
                
                restecg_val = st.selectbox(
                    FEATURE_INFO['restecg']['name'],
                    FEATURE_INFO['restecg']['options'],
                    help=FEATURE_INFO['restecg']['help']
                )
                inputs['restecg'] = FEATURE_INFO['restecg']['options'].index(restecg_val)
                
                exang_val = st.selectbox(
                    FEATURE_INFO['exang']['name'],
                    FEATURE_INFO['exang']['options'],
                    help=FEATURE_INFO['exang']['help']
                )
                inputs['exang'] = 1 if exang_val == 'Yes' else 0
                
                inputs['oldpeak'] = st.slider(
                    FEATURE_INFO['oldpeak']['name'],
                    FEATURE_INFO['oldpeak']['min'],
                    FEATURE_INFO['oldpeak']['max'],
                    FEATURE_INFO['oldpeak']['default'],
                    step=0.1,
                    help=FEATURE_INFO['oldpeak']['help']
                )
            
            with c2:
                slope_val = st.selectbox(
                    FEATURE_INFO['slope']['name'],
                    FEATURE_INFO['slope']['options'],
                    help=FEATURE_INFO['slope']['help']
                )
                inputs['slope'] = FEATURE_INFO['slope']['options'].index(slope_val)
                
                ca_val = st.selectbox(
                    FEATURE_INFO['ca']['name'],
                    FEATURE_INFO['ca']['options'],
                    help=FEATURE_INFO['ca']['help']
                )
                inputs['ca'] = int(ca_val)
                
                thal_val = st.selectbox(
                    FEATURE_INFO['thal']['name'],
                    FEATURE_INFO['thal']['options'],
                    help=FEATURE_INFO['thal']['help']
                )
                inputs['thal'] = FEATURE_INFO['thal']['options'].index(thal_val) + 1
    
    with col2:
        st.header("🔍 Prediction")
        
        # Predict button
        if st.button("🚀 Predict Heart Disease Risk", type="primary", use_container_width=True):
            # Prepare features
            features = [
                inputs['age'], inputs['sex'], inputs['cp'], inputs['trestbps'],
                inputs['chol'], inputs['fbs'], inputs['restecg'], inputs['thalach'],
                inputs['exang'], inputs['oldpeak'], inputs['slope'], inputs['ca'],
                inputs['thal']
            ]
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Calculate risk score
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(features_scaled)[0]
                # Normalize to 0-100 range
                risk_score = min(max((decision + 2) * 25, 0), 100)
            else:
                risk_score = 50  # Default
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "Low"
                risk_color = "green"
                emoji = "✅"
            elif risk_score < 60:
                risk_level = "Moderate"
                risk_color = "orange"
                emoji = "⚠️"
            else:
                risk_level = "High"
                risk_color = "red"
                emoji = "🚨"
            
            # Display results
            st.markdown("---")
            
            if prediction == 1:
                st.error(f"{emoji} **Disease Detected**")
            else:
                st.success(f"{emoji} **No Disease Detected**")
            
            # Risk metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Level", risk_level)
            with col_b:
                st.metric("Risk Score", f"{risk_score:.1f}%")
            
            # Gauge chart
            st.plotly_chart(
                create_gauge_chart(risk_score, "Risk Assessment"),
                use_container_width=True
            )
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if risk_level == "High":
                st.warning("""
                **High Risk Detected**
                - Consult a cardiologist immediately
                - Consider comprehensive cardiac evaluation
                - Review medication and lifestyle
                - Monitor symptoms closely
                """)
            elif risk_level == "Moderate":
                st.info("""
                **Moderate Risk**
                - Schedule regular check-ups
                - Maintain healthy lifestyle
                - Monitor blood pressure and cholesterol
                - Consider stress management
                """)
            else:
                st.success("""
                **Low Risk**
                - Continue healthy habits
                - Regular annual check-ups
                - Maintain exercise routine
                - Balanced diet
                """)
    
    # Footer
    st.markdown("---")
    
    # Expandable sections
    with st.expander("📚 How It Works"):
        st.markdown("""
        This system uses a **Support Vector Machine (SVM)** trained on 302 patient records 
        to predict heart disease risk. The model analyzes 13 clinical features including:
        
        - Patient demographics (age, sex)
        - Clinical measurements (blood pressure, cholesterol, blood sugar)
        - Cardiac assessments (ECG results, chest pain type, exercise tests)
        
        The system achieves **86.11% F1 Score** and **93.94% Recall**, meaning it's highly 
        effective at identifying patients who may have heart disease.
        """)
    
    with st.expander("⚠️ Important Disclaimer"):
        st.warning("""
        **Medical Disclaimer**
        
        This prediction system is designed for **research and educational purposes only**. 
        It should NOT be used as a substitute for professional medical advice, diagnosis, 
        or treatment.
        
        - Always seek the advice of qualified healthcare providers
        - Never disregard professional medical advice
        - Never delay seeking medical care based on information from this system
        - In case of emergency, call emergency services immediately
        
        This tool is meant to assist in understanding risk factors, not to provide 
        medical diagnosis.
        """)
    
    with st.expander("📊 Model Performance Details"):
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("""
            **Performance Metrics:**
            - Accuracy: 83.61%
            - Precision: 79.49%
            - Recall: 93.94%
            - F1 Score: 86.11%
            
            **Cross-Validation:**
            - Mean CV Score: 82.19%
            - Std Deviation: 5.40%
            """)
        
        with col_y:
            st.markdown("""
            **Model Details:**
            - Algorithm: Support Vector Machine
            - Kernel: RBF (Radial Basis Function)
            - Training Samples: 302
            - Features: 13 clinical parameters
            - Target: Binary (Disease/No Disease)
            """)

if __name__ == "__main__":
    main()
