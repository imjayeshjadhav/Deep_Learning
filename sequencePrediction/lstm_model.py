"""
LSTM-Based Sequence Prediction Model
Author: Jayesh Jadhav
PRN: 202301040019
Batch: DL1

LSTM Mathematical Model Explanation:
=====================================

1. FORGET GATE:
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   - Decides what information to discard from cell state
   - Uses sigmoid activation (0-1 range)

2. INPUT GATE:
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   - Decides what new information to store
   - Creates candidate values to add

3. CELL STATE UPDATE:
   C_t = f_t * C_{t-1} + i_t * C̃_t
   - Combines forget and input gates
   - Updates long-term memory

4. OUTPUT GATE:
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   - Controls what parts of cell state to output
   - Produces hidden state for current timestep

Where:
- σ = sigmoid function
- W = weight matrices
- b = bias vectors
- h = hidden state
- C = cell state
- x = input
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import os
import nltk
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMSequencePredictor:
    """
    LSTM-based sequence prediction model for next word prediction
    
    Mathematical Foundation:
    - Uses LSTM cells with forget, input, and output gates
    - Processes sequences of words to predict the next word
    - Employs embedding layer for word representation
    """
    
    def __init__(self, sequence_length=8, embedding_dim=64, lstm_units=128):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.max_vocab = 5000  # cap vocabulary to keep model fast
        
    def prepare_data(self, text_data):
        """
        Prepare text data for LSTM training
        
        Args:
            text_data (str): Raw text data
            
        Returns:
            tuple: (X, y) training sequences and targets
        """
        logger.info("Preparing data for LSTM training...")
        
        # Download NLTK data if needed
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        
        # Tokenize text into sentences
        sentences = nltk.sent_tokenize(text_data.lower())
        
        # Create tokenizer — cap at top 5000 words, no OOV token
        self.tokenizer = Tokenizer(num_words=self.max_vocab)
        self.tokenizer.fit_on_texts(sentences)
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_vocab)
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(sentences)
        
        # Create training sequences
        X, y = [], []
        for sequence in sequences:
            for i in range(self.sequence_length, len(sequence)):
                X.append(sequence[i-self.sequence_length:i])
                y.append(sequence[i])
        
        X = np.array(X)
        y = to_categorical(y, num_classes=self.vocab_size)
        
        logger.info(f"Created {len(X)} training sequences")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        return X, y
    
    def build_model(self):
        """
        Build LSTM model architecture
        
        Architecture:
        1. Embedding Layer: Converts word indices to dense vectors
        2. LSTM Layer: Processes sequences with memory gates
        3. Dropout Layer: Prevents overfitting
        4. Dense Layer: Output layer with softmax activation
        """
        logger.info("Building LSTM model...")
        
        self.model = Sequential([
            # Embedding layer for word representation
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length
            ),
            
            # LSTM layer with return_sequences=False for many-to-one
            LSTM(
                units=self.lstm_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=False
            ),
            
            # Dropout for regularization
            Dropout(0.3),
            
            # Output layer with softmax for word probability distribution
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X, y, epochs=50, batch_size=64, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            X (np.array): Input sequences
            y (np.array): Target words (one-hot encoded)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Validation data split ratio
        """
        logger.info("Starting LSTM training...")
        
        # Callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=256,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        return history
    
    def predict_next_word(self, text, top_k=5):
        """
        Predict next word(s) given input text
        
        Args:
            text (str): Input text sequence
            top_k (int): Number of top predictions to return
            
        Returns:
            list: Top k predicted words with probabilities
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained or loaded!")
        
        # Preprocess input text
        sequence = self.tokenizer.texts_to_sequences([text.lower()])[0]
        
        # Pad sequence to required length
        if len(sequence) >= self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        else:
            sequence = [0] * (self.sequence_length - len(sequence)) + sequence
        
        sequence = np.array([sequence])
        
        # Predict
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Convert indices to words, skip index 0
        word_index_reverse = {v: k for k, v in self.tokenizer.word_index.items()}

        # Get top k predictions, skipping unknown/padding indices
        sorted_indices = np.argsort(predictions)[::-1]
        results = []
        for idx in sorted_indices:
            if idx == 0:
                continue
            if idx in word_index_reverse:
                results.append({
                    'word': word_index_reverse[idx],
                    'probability': float(predictions[idx])
                })
            if len(results) == top_k:
                break

        return results
    
    def save_model(self, model_path='lstm_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save trained model and tokenizer"""
        if self.model:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        
        if self.tokenizer:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path='lstm_model.h5', tokenizer_path='tokenizer.pkl'):
        """Load trained model and tokenizer"""
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.vocab_size = len(self.tokenizer.word_index) + 1
            logger.info(f"Tokenizer loaded from {tokenizer_path}")

def create_sample_dataset():
    """
    Downloads WikiText-2 filtered to ML/AI articles, falls back to a
    rich built-in ML corpus so predictions are domain-relevant.
    """
    try:
        from datasets import load_dataset
        logger.info("Downloading WikiText-2 dataset...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        # Keep only paragraphs that mention ML/AI keywords
        keywords = {
            "learning", "neural", "network", "model", "training", "data",
            "algorithm", "classification", "regression", "prediction",
            "gradient", "loss", "layer", "weight", "feature", "input",
            "output", "function", "matrix", "vector", "probability"
        }
        lines = []
        for row in ds:
            t = row["text"].strip()
            if t and any(k in t.lower() for k in keywords):
                lines.append(t)
        text = " ".join(lines)[:300_000]
        if len(text) < 10_000:
            raise ValueError("Too little relevant text found")
        logger.info(f"Filtered WikiText-2 loaded: {len(text):,} characters")
        return text
    except Exception as e:
        logger.warning(f"Could not load WikiText-2 ({e}), using built-in corpus.")
        return _builtin_corpus()


def _builtin_corpus():
    base = (
        "machine learning is a method of data analysis that automates analytical model building. "
        "deep learning is part of a broader family of machine learning methods based on artificial neural networks. "
        "a neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data. "
        "the lstm network is a type of recurrent neural network used in deep learning. "
        "the forget gate of the lstm network decides what information to remove from the cell state. "
        "the input gate of the lstm network decides what new information to store in the cell state. "
        "the output gate of the lstm network decides what information to output from the cell state. "
        "gradient descent is an optimization algorithm used to minimize the loss function during training. "
        "backpropagation is the algorithm used to calculate gradients in a neural network. "
        "overfitting occurs when a neural network model learns the training data too well. "
        "dropout is a regularization technique that randomly drops neurons during training. "
        "the embedding layer converts word indices into dense vector representations. "
        "the softmax function converts the output of the neural network into a probability distribution. "
        "the loss function measures how well the neural network model fits the training data. "
        "the training data is used to train the neural network model parameters. "
        "the validation data is used to evaluate the neural network model during training. "
        "the test data is used to evaluate the final neural network model performance. "
        "natural language processing is a subfield of artificial intelligence and linguistics. "
        "sequence prediction is the task of predicting the next element in a sequence. "
        "word embeddings are dense vector representations of words in a continuous vector space. "
        "the attention mechanism allows the model to focus on relevant parts of the input sequence. "
        "transfer learning is a technique where a model trained on one task is reused on another task. "
        "convolutional neural networks are used for image recognition and classification tasks. "
        "recurrent neural networks are designed to work with sequential data and time series. "
        "the adam optimizer is an adaptive learning rate optimization algorithm for training neural networks. "
        "batch normalization is a technique to normalize the inputs of each layer in the neural network. "
        "the relu activation function is commonly used in hidden layers of neural networks. "
        "the sigmoid activation function outputs values between zero and one for binary classification. "
        "hyperparameter tuning is the process of finding the optimal settings for a machine learning model. "
        "cross validation is a technique to assess how well a model generalizes to independent data. "
    )
    return base * 60  # ~60k words, enough for meaningful training

if __name__ == "__main__":
    # Example usage
    predictor = LSTMSequencePredictor()
    
    # Create sample dataset
    text_data = create_sample_dataset()
    
    # Prepare data
    X, y = predictor.prepare_data(text_data)
    
    # Build and train model
    predictor.build_model()
    history = predictor.train(X, y, epochs=30)
    
    # Save model
    predictor.save_model()
    
    # Test prediction
    test_text = "machine learning is"
    predictions = predictor.predict_next_word(test_text)
    print(f"Next word predictions for '{test_text}':")
    for pred in predictions:
        print(f"  {pred['word']}: {pred['probability']:.4f}")