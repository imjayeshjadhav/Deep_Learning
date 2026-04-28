"""
FastAPI Application for LSTM Sequence Prediction
Author: Jayesh Jadhav
PRN: 202301040019
Batch: DL1

This FastAPI application provides REST API endpoints for:
1. Training LSTM model
2. Predicting next words
3. Model status and health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import logging
from datetime import datetime
import json

from lstm_model import LSTMSequencePredictor, create_sample_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LSTM Sequence Prediction API",
    description="AI-powered next word prediction using LSTM neural networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor = LSTMSequencePredictor()
model_status = {
    "is_trained": False,
    "last_trained": None,
    "vocab_size": 0,
    "training_in_progress": False
}

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5

class PredictionResponse(BaseModel):
    input_text: str
    predictions: List[dict]
    timestamp: str

class TrainingRequest(BaseModel):
    text_data: Optional[str] = None
    epochs: Optional[int] = 30
    sequence_length: Optional[int] = 10

class TrainingResponse(BaseModel):
    message: str
    status: str
    timestamp: str

class ModelStatus(BaseModel):
    is_trained: bool
    last_trained: Optional[str]
    vocab_size: int
    training_in_progress: bool
    model_info: dict

@app.on_event("startup")
async def startup_event():
    """Load existing model if available"""
    global predictor, model_status
    
    try:
        if os.path.exists("lstm_model.h5") and os.path.exists("tokenizer.pkl"):
            predictor.load_model()
            model_status["is_trained"] = True
            model_status["vocab_size"] = predictor.vocab_size
            logger.info("Existing model loaded successfully")
        else:
            logger.info("No existing model found. Ready for training.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive MVP UI for LSTM Sequence Prediction"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LSTM Next Word Predictor</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 30px 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { font-size: 1.8rem; color: #7dd3fc; margin-bottom: 4px; }
            .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 30px; }

            .card { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #334155; }
            .card h2 { font-size: 1rem; color: #7dd3fc; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; }

            /* Status bar */
            .status-bar { display: flex; align-items: center; gap: 10px; font-size: 0.85rem; }
            .dot { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; flex-shrink: 0; }
            .dot.trained { background: #22c55e; }
            .dot.training { background: #f59e0b; animation: pulse 1s infinite; }
            @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

            /* Train section */
            textarea { width: 100%; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; padding: 12px; font-size: 0.9rem; resize: vertical; min-height: 100px; outline: none; }
            textarea:focus { border-color: #7dd3fc; }
            .row { display: flex; gap: 12px; margin-top: 12px; align-items: center; flex-wrap: wrap; }
            label { font-size: 0.85rem; color: #94a3b8; }
            input[type=number] { width: 80px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #e2e8f0; padding: 8px; font-size: 0.9rem; outline: none; }
            input[type=number]:focus { border-color: #7dd3fc; }
            button { background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 0.9rem; cursor: pointer; transition: background 0.2s; }
            button:hover { background: #2563eb; }
            button:disabled { background: #334155; color: #64748b; cursor: not-allowed; }
            .btn-secondary { background: #334155; }
            .btn-secondary:hover { background: #475569; }

            /* Predict section */
            .predict-input-wrap { display: flex; gap: 10px; }
            .predict-input-wrap input[type=text] { flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; padding: 12px; font-size: 1rem; outline: none; }
            .predict-input-wrap input[type=text]:focus { border-color: #7dd3fc; }

            /* Prediction results */
            .predictions { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 10px; }
            .pred-chip { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 16px; cursor: pointer; transition: all 0.15s; }
            .pred-chip:hover { border-color: #7dd3fc; background: #1e3a5f; }
            .pred-word { font-size: 1rem; font-weight: 600; color: #7dd3fc; }
            .pred-prob { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
            .pred-bar { height: 3px; background: #3b82f6; border-radius: 2px; margin-top: 6px; transition: width 0.3s; }

            /* Log */
            .log { background: #0f172a; border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #64748b; font-family: monospace; max-height: 120px; overflow-y: auto; margin-top: 12px; }
            .log .info { color: #22c55e; }
            .log .err { color: #ef4444; }

            .hint { font-size: 0.8rem; color: #475569; margin-top: 8px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>🧠 LSTM Next Word Predictor</h1>
        <!-- Status -->
        <div class="card">
            <h2>Model Status</h2>
            <div class="status-bar">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">Checking...</span>
            </div>
        </div>

        <!-- Train -->
        <div class="card">
            <h2>Train Model</h2>
            <textarea id="trainText" placeholder="Paste your training text here, or leave empty to use the built-in sample dataset about ML/AI..."></textarea>
            <div class="row">
                <label>Epochs <input type="number" id="epochs" value="30" min="5" max="200"></label>
                <button id="trainBtn" onclick="trainModel()">Train LSTM</button>
                <button class="btn-secondary" onclick="loadSample()">Load WikiText-2 📚</button>
            </div>
            <div class="log" id="trainLog">Ready.</div>
        </div>

        <!-- Predict -->
        <div class="card">
            <h2>Predict Next Word</h2>
            <div class="predict-input-wrap">
                <input type="text" id="predictInput" placeholder="Type a phrase e.g. machine learning is" onkeydown="if(event.key==='Enter') predict()">
                <button id="predictBtn" onclick="predict()">Predict →</button>
            </div>
            <p class="hint">Click a predicted word to append it and keep predicting.</p>
            <div class="predictions" id="predictions"></div>
        </div>
    </div>

    <script>
        async function checkStatus() {
            try {
                const r = await fetch('/status');
                const d = await r.json();
                const dot = document.getElementById('statusDot');
                const txt = document.getElementById('statusText');
                if (d.training_in_progress) {
                    dot.className = 'dot training';
                    txt.textContent = 'Training in progress...';
                    setTimeout(checkStatus, 2000);
                } else if (d.is_trained) {
                    dot.className = 'dot trained';
                    txt.textContent = `Model ready — Vocab size: ${d.vocab_size} words | Last trained: ${d.last_trained ? new Date(d.last_trained).toLocaleString() : 'N/A'}`;
                } else {
                    dot.className = 'dot';
                    txt.textContent = 'Model not trained yet. Train it below.';
                }
            } catch(e) { document.getElementById('statusText').textContent = 'Could not reach server.'; }
        }

        async function trainModel() {
            const text = document.getElementById('trainText').value.trim();
            const epochs = parseInt(document.getElementById('epochs').value);
            const btn = document.getElementById('trainBtn');
            const log = document.getElementById('trainLog');

            btn.disabled = true;
            log.innerHTML = '<span class="info">Starting training...</span>';

            const dot = document.getElementById('statusDot');
            dot.className = 'dot training';
            document.getElementById('statusText').textContent = 'Training in progress...';

            try {
                const r = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text_data: text || null, epochs })
                });
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail);
                log.innerHTML = `<span class="info">${d.message}</span>`;
                pollTraining(btn, log);
            } catch(e) {
                log.innerHTML = `<span class="err">Error: ${e.message}</span>`;
                btn.disabled = false;
                checkStatus();
            }
        }

        async function pollTraining(btn, log) {
            const r = await fetch('/status');
            const d = await r.json();
            if (d.training_in_progress) {
                log.innerHTML = '<span class="info">Training... (this may take a few minutes)</span>';
                setTimeout(() => pollTraining(btn, log), 3000);
            } else if (d.is_trained) {
                log.innerHTML = `<span class="info">✅ Training complete! Vocab: ${d.vocab_size} words.</span>`;
                btn.disabled = false;
                checkStatus();
            } else {
                log.innerHTML = '<span class="err">Training may have failed. Check server logs.</span>';
                btn.disabled = false;
                checkStatus();
            }
        }

        async function predict() {
            const text = document.getElementById('predictInput').value.trim();
            if (!text) return;
            const container = document.getElementById('predictions');
            container.innerHTML = '<span style="color:#64748b;font-size:0.85rem">Predicting...</span>';

            try {
                const r = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text, top_k: 6 })
                });
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail);

                const maxProb = d.predictions[0]?.probability || 1;
                container.innerHTML = d.predictions.map(p => `
                    <div class="pred-chip" onclick="appendWord('${p.word}')">
                        <div class="pred-word">${p.word}</div>
                        <div class="pred-prob">${(p.probability * 100).toFixed(1)}%</div>
                        <div class="pred-bar" style="width:${(p.probability/maxProb*100).toFixed(0)}%"></div>
                    </div>
                `).join('');
            } catch(e) {
                container.innerHTML = `<span style="color:#ef4444;font-size:0.85rem">${e.message}</span>`;
            }
        }

        function appendWord(word) {
            const input = document.getElementById('predictInput');
            input.value = (input.value.trim() + ' ' + word).trim();
            predict();
        }

        async function loadSample() {
            const ta = document.getElementById('trainText');
            ta.value = 'Loading WikiText-2 dataset (~2M words)...';
            ta.disabled = true;
            // Trigger train with null text_data so server fetches WikiText-2 itself
            ta.value = '';
            ta.placeholder = 'WikiText-2 will be auto-downloaded on the server when you click Train (leave this empty).';
            ta.disabled = false;
        }

        checkStatus();
    </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

@app.get("/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status and information"""
    global model_status, predictor
    
    model_info = {
        "sequence_length": predictor.sequence_length,
        "embedding_dim": predictor.embedding_dim,
        "lstm_units": predictor.lstm_units,
        "architecture": "Embedding -> LSTM -> Dropout -> Dense(Softmax)"
    }
    
    return ModelStatus(
        is_trained=model_status["is_trained"],
        last_trained=model_status["last_trained"],
        vocab_size=model_status["vocab_size"],
        training_in_progress=model_status["training_in_progress"],
        model_info=model_info
    )

async def train_model_background(text_data: str, epochs: int, sequence_length: int):
    """Background task for model training"""
    global predictor, model_status
    
    try:
        model_status["training_in_progress"] = True
        logger.info("Starting background model training...")
        
        # Create new predictor with specified parameters
        predictor = LSTMSequencePredictor(sequence_length=sequence_length)
        
        # Prepare data
        X, y = predictor.prepare_data(text_data)
        
        # Build and train model
        predictor.build_model()
        history = predictor.train(X, y, epochs=epochs)
        
        # Save model
        predictor.save_model()
        
        # Update status
        model_status["is_trained"] = True
        model_status["last_trained"] = datetime.now().isoformat()
        model_status["vocab_size"] = predictor.vocab_size
        model_status["training_in_progress"] = False
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        model_status["training_in_progress"] = False
        raise

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the LSTM model"""
    global model_status
    
    if model_status["training_in_progress"]:
        raise HTTPException(
            status_code=400,
            detail="Training already in progress. Please wait for completion."
        )
    
    # Use provided text or sample dataset
    text_data = request.text_data if request.text_data else create_sample_dataset()
    
    if len(text_data.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Text data too short. Please provide at least 100 characters."
        )
    
    # Start background training
    background_tasks.add_task(
        train_model_background,
        text_data,
        request.epochs,
        request.sequence_length
    )
    
    return TrainingResponse(
        message="Model training started in background. Check /status for progress.",
        status="training_started",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_words(request: PredictionRequest):
    """Predict next words for given text sequence"""
    global predictor, model_status
    
    if not model_status["is_trained"]:
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Please train the model first using /train endpoint."
        )
    
    if model_status["training_in_progress"]:
        raise HTTPException(
            status_code=400,
            detail="Model training in progress. Please wait for completion."
        )
    
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )
    
    try:
        # Get predictions
        predictions = predictor.predict_next_word(request.text, request.top_k)
        
        return PredictionResponse(
            input_text=request.text,
            predictions=predictions,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload text file for training data"""
    if not file.filename.endswith(('.txt', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .csv files are supported."
        )
    
    try:
        content = await file.read()
        text_data = content.decode('utf-8')
        
        if len(text_data.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file too short. Please provide at least 100 characters."
            )
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "size": len(text_data),
            "preview": text_data[:200] + "..." if len(text_data) > 200 else text_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )