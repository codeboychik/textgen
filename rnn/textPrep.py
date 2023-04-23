import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def correctText(input_text):
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.h5')

    # Tokenization
    nltk.download('punkt')
    tokens = nltk.word_tokenize(input_text)

    # Text cleaning
    cleaned_tokens = [token.lower() for token in tokens if token.isalpha()]

    # Calculate vocabulary size
    vocab_size = len(set(cleaned_tokens))

    # Numerical representation
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(cleaned_tokens)
    sequences = tokenizer.texts_to_sequences(cleaned_tokens)
    padded_sequences = pad_sequences(sequences, maxlen=300)

    # Use the trained model to predict the corrected probabilities for each word
    corrected_probs = model.predict(padded_sequences)

    # Find the word with the highest probability for each input word
    corrected_indices = np.argmax(corrected_probs, axis=1)
    corrected_words = [tokenizer.index_word[index] for index in corrected_indices]

    # Concatenate the corrected words to form the corrected sentence
    corrected_text = ' '.join(corrected_words)

    return corrected_text