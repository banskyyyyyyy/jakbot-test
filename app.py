from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load NLP resources
nltk.download('punkt_tab')
lemmatizer = WordNetLemmatizer()

# Load trained model and data
MODEL_DIR = 'model/'
model = load_model(f'{MODEL_DIR}model.h5')
intents = json.load(open(f'{MODEL_DIR}data.json'))
words = pickle.load(open(f'{MODEL_DIR}texts.pkl', 'rb'))
classes = pickle.load(open(f'{MODEL_DIR}labels.pkl', 'rb'))

# Proses Tokenize and lemmatize sentence    
def clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

# Proses membuat bag-of-words vector
def create_bow(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Proses memprediksi dengan confidence threshold
def predict_intent(sentence, confidence_threshold=0.7):
    bow_vector = create_bow(sentence, words)
    predictions = model.predict(np.array([bow_vector]))[0]
    best_index = np.argmax(predictions)
    confidence = predictions[best_index]
    print(f"[DEBUG] Intent: {classes[best_index]} | Confidence: {confidence:.2f}")
    return "unknown" if confidence < confidence_threshold else classes[best_index]

def get_response(intent):
    if intent == "unknown":
        return "Maaf, aku belum paham pertanyaan kamu. Coba tanyakan hal lain seputar Persija ya\n\nAtau bisa saya bantu menjawab pertanyaan seperti:\n\n👉Jadwal persija\n👉Klasemen Persija\n👉langkah-langkah membuat KTA"

    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i.get('responses', ["Maaf aku tidak bisa menjawab pertanyaan itu."]))

    return "Maaf aku tidak tau pertanyaan kamu, bisa tanyakan hal lain seputar Persija."


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_message = request.args.get('msg')
    intent = predict_intent(user_message)
    response = get_response(intent)
    return response.replace("\n", "<br>")

if __name__ == "__main__":
    app.run()
