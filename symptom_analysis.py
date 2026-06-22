# symptom_analysis.py: Analyze symptoms, format questions, parse answers, evaluate preliminary results

import re
from config import SYMPTOM_DATABASE, TOP_SYMPTOMS, DIABETES_FEATURES, HEART_FEATURES

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
        'diabetes': "Have you been diagnosed with diabetes? (Yes/No)",
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
            numbers = re.findall(r'\d+\.?\d*', answer)
            return float(numbers[0]) if numbers else 0.0
        except:
            return 0.0
    else:
        return 1.0 if any(word in answer_lower for word in ['yes', 'true', 'y', '1']) else 0.0

def evaluate_preliminary_results(preliminary_data):
    """Evaluate if preliminary symptoms indicate high likelihood for each disease (threshold: at least 1 yes)"""
    proceed_diseases = []
    
    if "diabetes" in preliminary_data:
        diabetes_score = sum(preliminary_data["diabetes"].values())
        if diabetes_score >= 1:
            proceed_diseases.append("diabetes")
    
    if "heart_disease" in preliminary_data:
        heart_score = sum(preliminary_data["heart_disease"].values())
        if heart_score >= 1:
            proceed_diseases.append("heart_disease")
    
    return proceed_diseases

def build_question_queue(proceed_diseases):
    """Build unique question queue for proceeding diseases"""
    unique_features = set()
    for disease in proceed_diseases:
        if disease == "diabetes":
            unique_features.update(DIABETES_FEATURES)
        elif disease == "heart_disease":
            unique_features.update(HEART_FEATURES)
    
    question_queue = [(None, feature) for feature in unique_features]
    return question_queue