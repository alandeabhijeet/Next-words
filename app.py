from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense

app = Flask(__name__)

model = None
tokenizer = None
max_len = 0

def train_model(data):
    global model, tokenizer, max_len
    data = data.split('. ')
    maxlen50_data = []
    for string in data:
        words = string.split()
        current_chunk = ""
        for word in words:
            if len(current_chunk.split()) + 1 > 50:
                maxlen50_data.append(current_chunk)
                current_chunk = word
            else:
                current_chunk += (" " + word) if current_chunk else word
        if current_chunk:
            maxlen50_data.append(current_chunk)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(maxlen50_data)
    nwords = len(tokenizer.word_index)

    input_sequences = []
    for sen in maxlen50_data:
        tokenized_sentence = tokenizer.texts_to_sequences([sen])[0]
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i + 1])

    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]
    y = to_categorical(y, num_classes=nwords + 1)

    model = Sequential()
    model.add(Embedding(nwords + 1, 100, input_length=X.shape[1]))
    model.add(LSTM(150))
    model.add(Dense(nwords + 1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=70, verbose=1)
    model.save('text_generation_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

@app.route('/')
def home():
    return redirect(url_for('train_page'))

@app.route('/train', methods=['GET', 'POST'])
def train_page():
    if request.method == 'POST':
        data = request.form.get('data', '')
        if data:
            train_model(data)
            return redirect(url_for('generate_page'))
        else:
            return render_template('train.html', error="Please provide training data")
    return render_template('train.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate_page():
    global model, tokenizer, max_len

    if model is None or tokenizer is None:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        model = load_model('text_generation_model.h5')
        max_len = model.input_shape[1] + 1

    if request.method == 'POST':
        text = request.form.get('text', '')
        num_words = int(request.form.get('num_words', 10))
        generated_text = text

        for _ in range(num_words):
            token_text = tokenizer.texts_to_sequences([generated_text])[0]
            padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
            predicted_probabilities = model.predict(padded_token_text, verbose=0)
            predicted_word_index = np.argmax(predicted_probabilities, axis=-1)[0]
            word = next((w for w, idx in tokenizer.word_index.items() if idx == predicted_word_index), None)
            if word is None:
                break
            generated_text += ' ' + word

        return render_template('generate.html', generated_text=generated_text)
    return render_template('generate.html')

if __name__ == '__main__':
    app.run(debug=True)
