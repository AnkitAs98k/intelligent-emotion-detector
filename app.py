import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Define UI Page Configuration
st.set_page_config(page_title="Emotion Classifier", page_icon="🎭")
st.title("🎭 Text Emotion Classifier")
st.write("Enter some text below, and the LSTM model will predict the underlying emotion.")

# 2. Load the Model and Tokenizer
# We use @st.cache_resource so these large files only load once when the app starts
@st.cache_resource
def load_ml_assets():
    model = load_model("Artifacts/lstm_model.keras")
    with open("Artifacts/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

try:
    model, tokenizer = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model or tokenizer. Please ensure the 'Artifacts' folder exists. Details: {e}")
    st.stop()

# 3. Define the Label Mapping
# Based on the dair-ai/emotion dataset classes
emotion_labels = {
    0: ("Sadness", "😢"),
    1: ("Joy", "😂"),
    2: ("Love", "❤️"),
    3: ("Anger", "😡"),
    4: ("Fear", "😨"),
    5: ("Surprise", "😲")
}

# 4. User Input
user_input = st.text_area("What's on your mind?", placeholder="I am thrilled with the way my skin and hair feel...", height=150)

# 5. Prediction Logic
if st.button("Predict Emotion", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing text..."):
            # Preprocess the text exactly as done in training
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
            
            # Make prediction
            prediction = model.predict(padded_sequence)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            
            # Fetch label and emoji
            emotion_name, emoji = emotion_labels.get(predicted_class, ("Unknown", "❓"))
            
            # Display Results
            st.success("Analysis Complete!")
            st.markdown(f"### Predicted Emotion: {emotion_name} {emoji}")
            st.progress(int(confidence), text=f"Confidence: {confidence:.2f}%")