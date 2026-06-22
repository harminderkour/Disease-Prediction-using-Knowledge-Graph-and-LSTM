import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json
import joblib
import groq
import re

# ------------------------------
# 1️⃣ Load LSTM Models & Scalers
# ------------------------------
try:
    model_diabetes = load_model("ML/lstm_diabetes_model.h5")
except:
    model_diabetes = None

try:
    model_heart = load_model("ML/lstm_heart_model.h5")
except:
    model_heart = None

try:
    scaler_d = joblib.load("ML/scaler_diabetes.pkl")
    scaler_h = joblib.load("ML/scaler_heart.pkl")
except:
    scaler_d = StandardScaler()
    scaler_h = StandardScaler()

# ------------------------------
# 2️⃣ Define Required Features
# ------------------------------
diabetes_features = ['age','polyuria','polydipsia','sudden_weight_loss',
                     'weakness','polyphagia','obesity','alopecia']

heart_features = ['age', 'anaemia', 'creatinine_phosphokinase', 
                  'diabetes', 'ejection_fraction', 'high_blood_pressure', 
                  'platelets', 'serum_creatinine']

# ------------------------------
# 3️⃣ Groq API Setup
# ------------------------------
GROQ_API_KEY = "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"

def get_medical_response(user_message):
    """Get medical-focused response using Groq API"""
    try:
        client = groq.Client(api_key=GROQ_API_KEY)
        
        medical_prompt = f"""You are Dr. AI, a professional medical assistant specializing in diabetes and heart health. 
        Provide concise, helpful medical advice. If the question is not medical, politely redirect to health topics.
        
        Patient: {user_message}
        
        Dr. AI:"""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": medical_prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I'm here to help with medical questions about diabetes and heart health. How can I assist you today? (Error: {str(e)})"

# ------------------------------
# 4️⃣ Preprocessing Functions
# ------------------------------
def preprocess_diabetes(user_input_dict):
    X = [user_input_dict.get(f, 0) for f in diabetes_features]
    X = np.array(X).reshape(1, -1)
    X_scaled = scaler_d.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_lstm

def preprocess_heart(user_input_dict):
    X = [user_input_dict.get(f, 0) for f in heart_features]
    X = np.array(X).reshape(1, -1)
    X_scaled = scaler_h.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_lstm

# ------------------------------
# 5️⃣ Prediction Functions
# ------------------------------
def predict_diabetes(user_input_dict):
    if model_diabetes is None:
        return "Model unavailable", 0.0
    
    X = preprocess_diabetes(user_input_dict)
    pred_probs = model_diabetes.predict(X, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = float(pred_probs[0][pred_class])
    return "Positive" if pred_class == 1 else "Negative", confidence

def predict_heart(user_input_dict):
    if model_heart is None:
        return "Model unavailable", 0.0
    
    X = preprocess_heart(user_input_dict)
    pred_probs = model_heart.predict(X, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = float(pred_probs[0][pred_class])
    return "High Risk" if pred_class == 1 else "Low Risk", confidence

# ------------------------------
# 6️⃣ Streamlit Interface with Fixed Chat Handling
# ------------------------------
st.set_page_config(page_title="Medical Assistant AI", layout="wide", page_icon="🏥")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-bottom-right-radius: 5px;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-bottom-left-radius: 5px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 Medical Assistant AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your personal healthcare assistant for diabetes and heart health</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.waiting_for_response = False

# Sidebar for navigation
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choose Mode:", 
                       ["💬 Medical Chat", "🩺 Diabetes Assessment", "❤️ Heart Assessment"])
    
    st.markdown("---")
    st.header("📋 Quick Guide")
    
    if app_mode == "💬 Medical Chat":
        st.info("""
        **Ask about:**
        - Diabetes symptoms
        - Heart health
        - Medical advice
        - Health concerns
        """)
    elif app_mode == "🩺 Diabetes Assessment":
        st.warning("""
        **Provide:**
        - Age
        - Symptoms (Yes/No)
        - Health metrics
        """)
    else:
        st.warning("""
        **Provide:**
        - Age and health metrics
        - Medical history
        - Test results
        """)
    
    st.markdown("---")
    st.header("⚠️ Important")
    st.error("""
    This is not medical diagnosis.
    Consult healthcare professionals
    for proper medical advice.
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Dr. AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.waiting_for_response = True
        
        # Process the response
        if app_mode == "💬 Medical Chat":
            response = get_medical_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.waiting_for_response = False
            st.rerun()
        elif "predict" in prompt.lower():
            st.info("Please use the prediction forms in the sidebar for assessments")
            st.session_state.waiting_for_response = False

# Show loading indicator if waiting for response
if st.session_state.get('waiting_for_response', False):
    with col1:
        with st.spinner("Dr. AI is thinking..."):
            pass

with col2:
    if app_mode == "🩺 Diabetes Assessment":
        st.header("Diabetes Assessment")
        with st.form("diabetes_form"):
            user_input_dict = {}
            for feature in diabetes_features:
                if feature == 'age':
                    user_input_dict[feature] = st.number_input("Age", min_value=0, max_value=120, value=45)
                else:
                    user_input_dict[feature] = 1 if st.radio(f"{feature.replace('_', ' ').title()}", ["No", "Yes"]) == "Yes" else 0
            
            if st.form_submit_button("🔍 Assess Diabetes Risk"):
                result, confidence = predict_diabetes(user_input_dict)
                st.markdown(f'<div class="prediction-card"><h4>Result: {result}</h4><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
    
    elif app_mode == "❤️ Heart Assessment":
        st.header("Heart Assessment")
        with st.form("heart_form"):
            user_input_dict = {}
            for feature in heart_features:
                if feature in ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine']:
                    user_input_dict[feature] = st.number_input(feature.replace('_', ' ').title(), value=0.0)
                else:
                    user_input_dict[feature] = 1 if st.radio(f"{feature.replace('_', ' ').title()}", ["No", "Yes"]) == "Yes" else 0
            
            if st.form_submit_button("🔍 Assess Heart Risk"):
                result, confidence = predict_heart(user_input_dict)
                st.markdown(f'<div class="prediction-card"><h4>Result: {result}</h4><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.waiting_for_response = False
    st.rerun()