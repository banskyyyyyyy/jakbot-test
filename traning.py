import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import os

# ✅ Atur lokasi penyimpanan nltk_data di Railway
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# ✅ Download yang diperlukan
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

# NLP setup
lemmatizer = WordNetLemmatizer()

# Config path
MODEL_DIR = 'model/'
INTENTS_FILE = f'{MODEL_DIR}data.json'
TEXTS_PATH = f'{MODEL_DIR}texts.pkl'
LABELS_PATH = f'{MODEL_DIR}labels.pkl'
MODEL_PATH = f'{MODEL_DIR}model.h5'
IGNORE_WORDS = ['?', '!', '.', ':', ';', ',']

def load_intents_data():
    with open(INTENTS_FILE) as file:
        return json.load(file)

def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in IGNORE_WORDS]
    return sorted(set(words)), sorted(set(classes)), documents

def create_training_data(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = [0] * len(words)
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for w in pattern_words:
            if w in words:
                bag[words.index(w)] = 1
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    return np.array([x[0] for x in training]), np.array([x[1] for x in training])

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(256, input_shape=input_shape, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(
        optimizer=SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model():
    intents = load_intents_data()
    words, classes, documents = preprocess_data(intents)
    pickle.dump(words, open(TEXTS_PATH, 'wb'))
    pickle.dump(classes, open(LABELS_PATH, 'wb'))
    train_x, train_y = create_training_data(words, classes, documents)
    model = build_model((len(train_x[0]),), len(train_y[0]))
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save(MODEL_PATH)
    with open(f"{MODEL_DIR}history.pkl", 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Model training complete. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
