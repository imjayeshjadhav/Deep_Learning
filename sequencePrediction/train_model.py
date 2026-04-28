"""
Standalone LSTM Model Training Script
Author: Jayesh Jadhav
PRN: 202301040019
Batch: DL1

This script trains the LSTM model independently of the FastAPI application.
Useful for batch training with larger datasets.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from lstm_model import LSTMSequencePredictor, create_sample_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Training history plot saved as 'training_history.png'")

def load_text_file(file_path):
    """Load text data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train LSTM Sequence Prediction Model')
    parser.add_argument('--data', type=str, help='Path to training text file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=10, help='Input sequence length')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--lstm_units', type=int, default=128, help='LSTM units')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    
    args = parser.parse_args()
    
    # Load training data
    if args.data:
        text_data = load_text_file(args.data)
        if not text_data:
            logger.error("Failed to load training data. Using sample dataset.")
            text_data = create_sample_dataset()
    else:
        logger.info("No data file provided. Using sample dataset.")
        text_data = create_sample_dataset()
    
    # Initialize predictor
    predictor = LSTMSequencePredictor(
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units
    )
    
    # Prepare data
    logger.info("Preparing training data...")
    X, y = predictor.prepare_data(text_data)
    
    # Build model
    logger.info("Building LSTM model...")
    predictor.build_model()
    
    # Train model
    logger.info(f"Training model for {args.epochs} epochs...")
    history = predictor.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    predictor.save_model()
    logger.info("Model saved successfully!")
    
    # Plot training history if requested
    if args.plot:
        plot_training_history(history)
    
    # Test predictions
    test_sequences = [
        "machine learning is",
        "deep learning uses",
        "natural language processing",
        "lstm networks are",
        "the model predicts"
    ]
    
    logger.info("\nTesting predictions:")
    for seq in test_sequences:
        try:
            predictions = predictor.predict_next_word(seq, top_k=3)
            logger.info(f"'{seq}' -> {[p['word'] for p in predictions]}")
        except Exception as e:
            logger.error(f"Prediction failed for '{seq}': {e}")

if __name__ == "__main__":
    main()