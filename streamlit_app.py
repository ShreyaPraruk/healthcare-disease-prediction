import streamlit as st
import pickle

# Load model + symptoms
model, symptoms = pickle.load(open('model/model.pkl', 'rb'))

st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Healthcare Disease Prediction System")
st.markdown("### Select at least 3 symptoms for better accuracy")

# Display-friendly names
display_symptoms = [s.replace('_', ' ').title() for s in symptoms]

# Input
selected_display = st.multiselect("Symptoms", display_symptoms)

# Convert back
selected_symptoms = [s.lower().replace(' ', '_') for s in selected_display]

# Create vector
input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# Predict
if st.button("🔍 Predict Disease"):

    if len(selected_symptoms) < 3:
        st.warning("⚠️ Please select at least 3 symptoms")
    
    else:
        probs = model.predict_proba([input_data])[0]
        top_indices = probs.argsort()[-3:][::-1]

        st.subheader("🧾 Most Probable Diseases:")

        for i in top_indices:
            st.write(f"👉 {model.classes_[i]} ({round(probs[i]*100,2)}%)")

        st.success("✅ Prediction generated using machine learning model")

        st.info("💡 This system predicts based on patterns, not medical diagnosis")