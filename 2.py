import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json
import joblib
import groq
import re
from gtts import gTTS
import io
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# ------------------------------
# 1️⃣ Load LSTM Models & Scalers
# ------------------------------
try:
    model_diabetes = load_model("ML/lstm_diabetes_model.h5")
    # st.success("✅ Diabetes model loaded successfully")
except Exception as e:
    # st.error(f"❌ Error loading diabetes model: {e}")
    model_diabetes = None

try:
    model_heart = load_model("ML/lstm_heart_model.h5")
    # st.success("✅ Heart model loaded successfully")
except Exception as e:
    # st.error(f"❌ Error loading heart model: {e}")
    model_heart = None

try:
    scaler_d = joblib.load("ML/scaler_diabetes.pkl")
    scaler_h = joblib.load("ML/scaler_heart.pkl")
    # st.success("✅ Scalers loaded successfully")
except Exception as e:
    # st.error(f"❌ Error loading scalers: {e}")
    scaler_d = StandardScaler()
    scaler_h = StandardScaler()

# ------------------------------
# 2️⃣ Define LSTM Input Features
# ------------------------------
DIABETES_FEATURES = ['age', 'polyuria', 'polydipsia', 'sudden_weight_loss',
                     'weakness', 'polyphagia', 'obesity', 'alopecia']

HEART_FEATURES = ['age', 'anaemia', 'creatinine_phosphokinase', 
                  'diabetes', 'ejection_fraction', 'high_blood_pressure', 
                  'platelets', 'serum_creatinine']

# ------------------------------
# 3️⃣ Symptom Database for Detection
# ------------------------------
SYMPTOM_DATABASE = {
    "diabetes": [
        "excessive_hunger", "fatigue", "restlessness", "blurred_vision",
        "increased_appetite", "obesity", "polyuria", "weight_loss",
        "lethargy", "irregular_sugar", "thirst", "frequent_urination",
        "polydipsia", "sudden_weight_loss", "weakness", "polyphagia", "alopecia", "restless", 
    ],
    "heart_disease": [
        "chest_pain", "chest_pressure", "shortness_breath", "fatigue",
        "dizziness", "lightheadedness", "swelling_feet", "swelling_ankles",
        "vomiting", "sweating", "breathlessness", "palpitations",
        "anaemia", "high_blood_pressure", "creatinine_phosphokinase"
    ]
}


TOP_SYMPTOMS = {
    "diabetes": ['polyuria', 'polydipsia'],
    "heart_disease": ['chest_pain', 'shortness_breath']
}

# ------------------------------
# 5️⃣ Groq API Setup for Conversation
# ------------------------------
GROQ_API_KEY = "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"

def get_medical_response(user_message, conversation_context="", limit_suggestions=False):
    """Get medical-focused response using Groq API"""
    try:
        client = groq.Client(api_key=GROQ_API_KEY)
        
        if limit_suggestions:
            medical_prompt = f"""You are AI Health Chatbot, a professional medical assistant. 
            {conversation_context}
            Provide concise, helpful medical advice with EXACTLY 2 practical suggestions only. Keep responses professional and brief. Do not provide more than 2 suggestions.
            
            Patient: {user_message}
            
            Health Assistant :"""
        else:
            medical_prompt = f"""You are AI Health Chatbot, a professional medical assistant. 
            {conversation_context}
            Provide concise, helpful medical advice. Keep responses professional.
            
            Patient: {user_message}
            
            Health Assistant :"""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": medical_prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        if limit_suggestions:
            return "I'm here to help. Here are 2 suggestions: 1. Consult a healthcare professional for personalized advice. 2. Monitor your symptoms and rest adequately."
        return "I'm here to help with medical questions. How can I assist you today?"

# ------------------------------
# 6️⃣ Symptom Analysis & Question Management
# ------------------------------
def analyze_symptoms(text):
    """Analyze text to detect which diseases are likely"""
    text_lower = text.lower()
    
    detected_diseases = {"diabetes": False, "heart_disease": False}
    detected_symptoms = {"diabetes": [], "heart_disease": []}
    
    # Check for diabetes symptoms
    for symptom in SYMPTOM_DATABASE["diabetes"]:
        symptom_pattern = symptom.replace('_', '[-\\s]*')
        if re.search(symptom_pattern, text_lower):
            detected_diseases["diabetes"] = True
            detected_symptoms["diabetes"].append(symptom.replace('_', ' '))
    
    # Check for heart disease symptoms
    for symptom in SYMPTOM_DATABASE["heart_disease"]:
        symptom_pattern = symptom.replace('_', '[-\\s]*')
        if re.search(symptom_pattern, text_lower):
            detected_diseases["heart_disease"] = True
            detected_symptoms["heart_disease"].append(symptom.replace('_', ' '))
    
    return detected_diseases, detected_symptoms

def format_question(feature):
    """Format questions in a user-friendly way"""
    question_map = {
        'age': "What is your age?",
        'polyuria': "Do you experience excessive urination? (Yes/No)",
        'polydipsia': "Do you experience excessive thirst? (Yes/No)",
        'sudden_weight_loss': "Have you experienced sudden weight loss? (Yes/No)",
        'weakness': "Do you feel general weakness? (Yes/No)",
        'polyphagia': "Do you experience excessive hunger? (Yes/No)",
        'obesity': "Are you obese? (Yes/No)",
        'alopecia': "Do you experience hair loss? (Yes/No)",
        'anaemia': "Do you have anaemia? (Yes/No)",
        'creatinine_phosphokinase': "What is your creatinine phosphokinase level?",
        'diabetes': "Have you been diagonsed with diabetes? (Yes/No)",
        'ejection_fraction': "What is your ejection fraction percentage?",
        'high_blood_pressure': "Do you have high blood pressure? (Yes/No)",
        'platelets': "What is your platelets count?",
        'serum_creatinine': "What is your serum creatinine level?",
        'chest_pain': "Do you experience chest pain? (Yes/No)",
        'shortness_breath': "Do you experience shortness of breath? (Yes/No)"
    }
    return question_map.get(feature, f"Do you experience {feature.replace('_', ' ')}? (Yes/No)")

def parse_answer(feature, answer):
    """Parse user answers into appropriate formats"""
    answer_lower = answer.lower()
    
    if any(keyword in feature for keyword in ['age', 'creatinine', 'ejection', 'platelets', 'serum']):
        try:
            # Extract numbers from text
            numbers = re.findall(r'\d+\.?\d*', answer)
            return float(numbers[0]) if numbers else 0.0
        except:
            return 0.0
    else:
        # Boolean features (including preliminary symptoms)
        return 1.0 if any(word in answer_lower for word in ['yes', 'true', 'y', '1']) else 0.0

def evaluate_preliminary_results(preliminary_data):
    """Evaluate if preliminary symptoms indicate high likelihood for each disease (threshold: at least 1 yes)"""
    proceed_diseases = []
    
    if "diabetes" in preliminary_data:
        diabetes_score = sum(preliminary_data["diabetes"].values())
        if diabetes_score >= 1:  # At least 1 yes out of 2
            proceed_diseases.append("diabetes")
    
    if "heart_disease" in preliminary_data:
        heart_score = sum(preliminary_data["heart_disease"].values())
        if heart_score >= 1:  # At least 1 yes out of 2
            proceed_diseases.append("heart_disease")
    
    return proceed_diseases

# ------------------------------
# 7️⃣ Prediction Functions
# ------------------------------
def predict_diabetes(user_input_dict):
    if model_diabetes is None:
        return "Model unavailable", 0.0
    
    # Ensure all features are present
    complete_input = {feature: user_input_dict.get(feature, 0.0) for feature in DIABETES_FEATURES}
    
    X = np.array([[complete_input[f] for f in DIABETES_FEATURES]])
    X_scaled = scaler_d.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    pred_probs = model_diabetes.predict(X_lstm, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = float(pred_probs[0][pred_class])
    
    return "Positive" if pred_class == 1 else "Negative", confidence

def predict_heart(user_input_dict):
    if model_heart is None:
        return "Model unavailable", 0.0
    
    # Ensure all features are present
    complete_input = {feature: user_input_dict.get(feature, 0.0) for feature in HEART_FEATURES}
    
    X = np.array([[complete_input[f] for f in HEART_FEATURES]])
    X_scaled = scaler_h.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    pred_probs = model_heart.predict(X_lstm, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = float(pred_probs[0][pred_class])
    
    return "High Risk" if pred_class == 1 else "Low Risk", confidence

# ------------------------------
# 8️⃣ Streamlit Application
# ------------------------------
# ------------------------------
# 8️⃣ Streamlit Application
# ------------------------------
st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide", page_icon="🤖")

# Custom CSS with blue-white gradients and transitions
st.markdown("""
<style>
    /* Global body background with subtle blue-white gradient */
    .stApp {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 50%, #FFFFFF 100%);
        transition: background 0.3s ease-in-out;
    }
    
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 10px;
        border-radius: 10px;
        transition: transform 0.3s ease;
    }
    .main-header:hover {
        transform: scale(1.02);
    }
    
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-bottom-right-radius: 5px;
        color: #1976D2;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F5F5 100%);
        border-bottom-left-radius: 5px;
        color: #1565C0;
    }
    
    .question-highlight {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
        transition: box-shadow 0.3s ease;
    }
    .question-highlight:hover {
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-2px);
    }
    
    /* Chat input styling */
    .stChatInput input {
        border: 2px solid #2196F3;
        border-radius: 20px;
        padding: 10px 15px;
        transition: border-color 0.3s ease;
    }
    .stChatInput input:focus {
        border-color: #1976D2;
        box-shadow: 0 0 10px rgba(33, 150, 243, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header and Layout with Robot on Left
col1, col2 = st.columns([1, 4])  # Left: 20% for robot, Right: 80% for content

import base64
import streamlit as st

file = open("bot.jpg", "rb").read()
data_url = "data:image/jpg;base64," + base64.b64encode(file).decode()

with col1:
    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            text-align: left;
        ">
            <img src="{data_url}" width="500" height="500" style="border-radius: 20px; margin-bottom: 10px;">
            <p style="text-align: left; margin: 0;">🤖 <b> AI Assistant</b></p>
        </div>
        <hr>
        """,
        unsafe_allow_html=True
    )

with col1:
    st.markdown("#")  
    # st.image("graident-ai-robot-vectorart_78370-4114.jpg", width=150, caption="🤖 Dr. AI Assistant")  # Placeholder robot icon; use a healthcare robot URL
    # st.image(data_url, width=200, caption="🤖 Dr. AI Assistant")
    st.markdown("---")  

with col2:
    # Header (now in right column)
    st.markdown('<h1 class="main-header">🏥 Medical Diagnostic Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state (updated with proceed_diseases)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "diagnosis_mode" not in st.session_state:
        st.session_state.diagnosis_mode = False
    if "preliminary_mode" not in st.session_state:
        st.session_state.preliminary_mode = False
    if "detected_diseases" not in st.session_state:
        st.session_state.detected_diseases = {}
    if "proceed_diseases" not in st.session_state:  # NEW: For filtered predictions
        st.session_state.proceed_diseases = []
    if "collected_data" not in st.session_state:
        st.session_state.collected_data = {}
    if "preliminary_data" not in st.session_state:
        st.session_state.preliminary_data = {}
    if "preliminary_queue" not in st.session_state:
        st.session_state.preliminary_queue = []
    if "question_queue" not in st.session_state:
        st.session_state.question_queue = []

    # Display chat messages (now in right column) - FIXED: class typo for user messages
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Assistant message with voice button
            col_msg, col_btn = st.columns([4, 1])
            with col_msg:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Dr. AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            with col_btn:
                if st.button("🔊", key=f"voice_{index}"):
                    tts = gTTS(text=message["content"], lang='en')
                    audio_file = io.BytesIO()
                    tts.write_to_fp(audio_file)
                    audio_file.seek(0)
                    st.audio(audio_file, format='audio/mp3')

    # Handle preliminary questions
    if st.session_state.preliminary_mode and st.session_state.preliminary_queue:
        current_disease, current_feature = st.session_state.preliminary_queue[0]
        question_text = format_question(current_feature)
        
        st.markdown(f'<div class="question-highlight"><strong>Clarifying Question:</strong> {question_text}</div>', unsafe_allow_html=True)
        
        if answer := st.chat_input("Your answer...", key="preliminary_input"):
            # Parse and store answer
            parsed_answer = parse_answer(current_feature, answer)
            if current_disease not in st.session_state.preliminary_data:
                st.session_state.preliminary_data[current_disease] = {}
            st.session_state.preliminary_data[current_disease][current_feature] = parsed_answer
            
            # Add to chat
            st.session_state.messages.append({"role": "user", "content": answer})
            
            # Remove answered question
            st.session_state.preliminary_queue.pop(0)
            
            # Check if we have more preliminary questions
            if st.session_state.preliminary_queue:
                st.rerun()
            else:
                # All preliminary questions answered, evaluate
                proceed_diseases = evaluate_preliminary_results(st.session_state.preliminary_data)
                
                if proceed_diseases:
                    # Proceed to full diagnosis for qualifying diseases
                    st.session_state.diagnosis_mode = True
                    st.session_state.collected_data = {}
                    st.session_state.question_queue = []
                    st.session_state.proceed_diseases = proceed_diseases  # NEW: For filtered predictions
                    
                    # Collect unique features across all proceeding diseases to avoid duplicates
                    unique_features = set()
                    for disease in proceed_diseases:
                        if disease == "diabetes":
                            unique_features.update(DIABETES_FEATURES)
                        else:
                            unique_features.update(HEART_FEATURES)
                    
                    # Build full question queue with unique features
                    for feature in unique_features:
                        st.session_state.question_queue.append((None, feature))
                    
                    # Inform user
                    diseases_str = " and ".join(proceed_diseases)
                    response = f"Based on your answers, {diseases_str} seems likely. Now I'll ask detailed questions for accurate assessment."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.preliminary_mode = False
                    st.rerun()
                else:
                    # No high likelihood, give general advice
                    response = "Based on your answers, diabetes and heart disease seem less likely at this time. This symptom might be related to other causes. Please describe more details or consult a doctor."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.preliminary_mode = False
                    st.session_state.preliminary_data = {}
                    st.rerun()

    # Handle full diagnosis questions
    elif st.session_state.diagnosis_mode and st.session_state.question_queue:
        current_disease, current_feature = st.session_state.question_queue[0]
        question_text = format_question(current_feature)
        
        st.markdown(f'<div class="question-highlight"><strong>Question:</strong> {question_text}</div>', unsafe_allow_html=True)
        
        if answer := st.chat_input("Your answer...", key="diagnosis_input"):
            # Parse and store answer
            parsed_answer = parse_answer(current_feature, answer)
            st.session_state.collected_data[current_feature] = parsed_answer
            
            # Add to chat
            st.session_state.messages.append({"role": "user", "content": answer})
            
            # Remove answered question
            st.session_state.question_queue.pop(0)
            
            # Check if we have more questions
            if st.session_state.question_queue:
                st.rerun()
            else:
                # All questions answered, make predictions - UPDATED: Use stored proceed_diseases
                proceed_diseases = st.session_state.proceed_diseases

                prediction_text = "**📊 Diagnosis Results:**\n\n"

                if "diabetes" in proceed_diseases:
                    diabetes_result, diabetes_conf = predict_diabetes(st.session_state.collected_data)
                    prediction_text += f"**Diabetes:** {diabetes_result} ({diabetes_conf:.1%} confidence)\n"

                if "heart_disease" in proceed_diseases:
                    heart_result, heart_conf = predict_heart(st.session_state.collected_data)
                    prediction_text += f"**Heart Disease:** {heart_result} ({heart_conf:.1%} confidence)\n"

                # Only add disclaimer if any predictions were made
                if proceed_diseases:
                    prediction_text += "\n*Please consult a healthcare professional for proper medical diagnosis.*"
                else:
                    prediction_text = "No specific risks detected based on your inputs. General health advice applies."

                # FIXED: Render Markdown directly in card; clean only for message storage (no HTML stripping needed)
                st.markdown(f'<div class="prediction-card">{prediction_text}</div>', unsafe_allow_html=True)
                clean_prediction = re.sub('<[^<]+?>', '', prediction_text)  # Only for chat history
                st.session_state.messages.append({"role": "assistant", "content": clean_prediction})
                st.session_state.diagnosis_mode = False
                st.session_state.proceed_diseases = []  # Reset for next use
                st.rerun()

    # Normal chat input (updated to ensure unrelated symptoms skip questions and give exactly 2 suggestions)
    else:
        if prompt := st.chat_input("Describe your symptoms or ask a question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Analyze for symptoms
            detected_diseases, detected_symptoms = analyze_symptoms(prompt)
            
            if any(detected_diseases.values()):
                # Symptoms detected - start preliminary mode (questions only for related diseases)
                st.session_state.preliminary_mode = True
                st.session_state.diagnosis_mode = False  # Ensure no overlap
                st.session_state.detected_diseases = detected_diseases
                st.session_state.preliminary_data = {}
                
                # Build preliminary question queue (2 top symptoms per detected disease)
                preliminary_queue = []
                if detected_diseases["diabetes"]:
                    for feature in TOP_SYMPTOMS["diabetes"]:
                        preliminary_queue.append(("diabetes", feature))
                if detected_diseases["heart_disease"]:
                    for feature in TOP_SYMPTOMS["heart_disease"]:
                        preliminary_queue.append(("heart_disease", feature))
                
                st.session_state.preliminary_queue = preliminary_queue
                
                # Inform user
                response = f"I detected symptoms possibly related to "
                diseases = []
                if detected_diseases["diabetes"]:
                    diseases.append("diabetes")
                if detected_diseases["heart_disease"]:
                    diseases.append("heart disease")
                
                response += " and ".join(diseases) + ". Let me ask a few clarifying questions to assess further."
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                # UPDATED: Unrelated or no symptoms detected - skip ALL questions, give exactly 2 suggestions only
                # Reset modes and data to prevent any carryover or unnecessary questions
                st.session_state.preliminary_mode = False
                st.session_state.diagnosis_mode = False
                st.session_state.preliminary_data = {}
                st.session_state.preliminary_queue = []
                st.session_state.question_queue = []
                st.session_state.detected_diseases = {"diabetes": False, "heart_disease": False}  # Reset detections
                st.session_state.proceed_diseases = []  # Reset predictions
                
                # Get limited response with exactly 2 suggestions (handles incomplete prompts gracefully)
                response = get_medical_response(prompt, limit_suggestions=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()