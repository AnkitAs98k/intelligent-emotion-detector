from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model and tokenizer globally so they only load once when the server starts
model = load_model("Artifacts/lstm_model.keras")
with open("Artifacts/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Label mapping based on your dataset
emotion_labels = {
    0: "Sadness 😢",
    1: "Joy 😂",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}

@app.route('/')
def home():
    # Serves the index.html file from your templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive data from the frontend
    data = request.json
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Please enter some text.'})
    
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    
    # Predict emotion
    prediction = model.predict(padded_sequence)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    
    # Send the result back to the frontend
    return jsonify({
        'emotion': emotion_labels.get(predicted_class, "Unknown"),
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)