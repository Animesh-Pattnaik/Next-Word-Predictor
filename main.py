import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import time

# Initialize tokenizer
tokenizer = Tokenizer()

def tokenize_faqs(faqs):
    tokenizer.fit_on_texts([faqs])

    # Tokenize input sequences
    input_sequences = []
    for sentence in faqs.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])

    # Pad sequences
    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    return padded_input_sequences, tokenizer.word_index

def create_next_word_predictor_model(padded_input_sequences, vocab_size):
    # Prepare data
    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)

    # Define model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=X.shape[1]))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(X, y, epochs=50)

    return model

def main():
    st.title("Next Word Predictor")

    st.write(
        "This app allows you to upload a text file containing sentences and "
        "generates a next word predictor model based on the provided data. You can then use the model "
        "to predict the next word given some input words."
    )

    st.sidebar.title("Upload File")

    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload the Text File", type=["txt"])

    # Display file contents if uploaded
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")

        # Store content in variable
        faqs = content

        st.success("Content stored successfully!")

        # Tokenize FAQs
        padded_input_sequences, word_index = tokenize_faqs(faqs)

        # Create next word predictor model
        vocab_size = len(word_index) + 1
        model = create_next_word_predictor_model(padded_input_sequences, vocab_size)

        st.success("Next word predictor model created and trained successfully!")

    # Input field for some words
    st.header("Predict Next Word")
    text = st.text_input("Enter some words:", "")

    if text:
        st.success("Words stored successfully!")

        for i in range(10):
            # Tokenize
            token_text = tokenizer.texts_to_sequences([text])[0]
            # Padding
            padded_token_text = pad_sequences([token_text], maxlen=50, padding='pre')
            # Predict
            pos = np.argmax(model.predict(padded_token_text))

            for word, index in tokenizer.word_index.items():
                if index == pos:
                    text += " " + word
                    st.write(text)
                    time.sleep(1)

if __name__ == "__main__":
    main()
