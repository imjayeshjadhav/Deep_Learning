"""
Test Client for LSTM Sequence Prediction API
Author: Jayesh Jadhav
PRN: 202301040019
Batch: DL1

This script demonstrates how to interact with the FastAPI endpoints.
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMAPIClient:
    """Client for interacting with LSTM Sequence Prediction API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None
    
    def get_status(self):
        """Get model status"""
        try:
            response = requests.get(f"{self.base_url}/status")
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return None
    
    def train_model(self, text_data=None, epochs=30, sequence_length=10):
        """Train the model"""
        try:
            payload = {
                "epochs": epochs,
                "sequence_length": sequence_length
            }
            if text_data:
                payload["text_data"] = text_data
                
            response = requests.post(
                f"{self.base_url}/train",
                json=payload
            )
            return response.json()
        except Exception as e:
            logger.error(f"Training request failed: {e}")
            return None
    
    def predict_next_words(self, text, top_k=5):
        """Predict next words"""
        try:
            payload = {
                "text": text,
                "top_k": top_k
            }
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload
            )
            return response.json()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def wait_for_training(self, check_interval=10, max_wait=600):
        """Wait for training to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.get_status()
            if status:
                if not status.get("training_in_progress", False):
                    if status.get("is_trained", False):
                        logger.info("Training completed successfully!")
                        return True
                    else:
                        logger.error("Training failed!")
                        return False
                else:
                    logger.info("Training in progress... waiting...")
            
            time.sleep(check_interval)
        
        logger.error("Training timeout!")
        return False

def demo_api_usage():
    """Demonstrate API usage"""
    client = LSTMAPIClient()
    
    # 1. Health check
    logger.info("=== Health Check ===")
    health = client.health_check()
    if health:
        logger.info(f"API Status: {health['status']}")
    else:
        logger.error("API is not accessible!")
        return
    
    # 2. Check initial status
    logger.info("\n=== Initial Status ===")
    status = client.get_status()
    if status:
        logger.info(f"Model trained: {status['is_trained']}")
        logger.info(f"Vocab size: {status['vocab_size']}")
    
    # 3. Train model if not trained
    if not status.get("is_trained", False):
        logger.info("\n=== Training Model ===")
        
        # Custom training data (you can replace with your own)
        custom_data = """
        Artificial intelligence is transforming the world in unprecedented ways.
        Machine learning algorithms can learn patterns from data without explicit programming.
        Deep learning uses neural networks with multiple layers to solve complex problems.
        Natural language processing enables computers to understand and generate human language.
        Computer vision allows machines to interpret and analyze visual information.
        Reinforcement learning trains agents to make decisions through trial and error.
        Data science combines statistics, programming, and domain expertise to extract insights.
        Big data technologies handle massive volumes of structured and unstructured information.
        Cloud computing provides scalable and flexible infrastructure for modern applications.
        Internet of Things connects everyday objects to the digital world.
        Blockchain technology ensures secure and transparent transactions.
        Cybersecurity protects digital assets from threats and vulnerabilities.
        Software engineering principles guide the development of robust applications.
        Agile methodology promotes iterative and collaborative software development.
        DevOps practices integrate development and operations for faster delivery.
        """
        
        train_result = client.train_model(
            text_data=custom_data,
            epochs=25,
            sequence_length=8
        )
        
        if train_result:
            logger.info(f"Training started: {train_result['message']}")
            
            # Wait for training to complete
            if client.wait_for_training():
                logger.info("Training completed!")
            else:
                logger.error("Training failed or timed out!")
                return
        else:
            logger.error("Failed to start training!")
            return
    
    # 4. Test predictions
    logger.info("\n=== Testing Predictions ===")
    
    test_sequences = [
        "machine learning algorithms",
        "deep learning uses",
        "natural language processing",
        "artificial intelligence is",
        "data science combines",
        "neural networks with",
        "computer vision allows"
    ]
    
    for seq in test_sequences:
        result = client.predict_next_words(seq, top_k=3)
        if result:
            logger.info(f"\nInput: '{seq}'")
            logger.info("Predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                logger.info(f"  {i}. {pred['word']} (prob: {pred['probability']:.4f})")
        else:
            logger.error(f"Prediction failed for: {seq}")
    
    # 5. Final status
    logger.info("\n=== Final Status ===")
    final_status = client.get_status()
    if final_status:
        logger.info(f"Model Info: {json.dumps(final_status['model_info'], indent=2)}")

if __name__ == "__main__":
    logger.info("LSTM Sequence Prediction API Test Client")
    logger.info("Make sure the FastAPI server is running on http://localhost:8000")
    logger.info("Start server with: python main.py")
    
    input("\nPress Enter to start testing...")
    demo_api_usage()