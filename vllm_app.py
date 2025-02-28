import argparse
from pathlib import Path
import os
import logging
from typing import List, Dict, Any

import torch
from vllm import LLM, SamplingParams
from flask import Flask, render_template, request, jsonify, session
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.urandom(24)  # For session management


class VLLMApp:
    def __init__(self, model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=None):
        """Initialize the application with the specified model parameters."""
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None
        self.chat_histories = {}  # Store chat histories by session ID
        
    def initialize_model(self):
        """Initialize the model using vLLM."""
        logger.info(f"Initializing model from {self.model_path}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"Current GPU: {torch.cuda.current_device()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            
        if self.max_model_len:
            logger.info(f"Using max model length: {self.max_model_len}")
        
        try:
            # Initialize vLLM with the max_model_len parameter if provided
            if self.max_model_len:
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True
                )
            else:
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    trust_remote_code=True
                )
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def format_chat_history(self, history, context_window):
        """Format chat history for the model prompt."""
        # Limit history to the last 'context_window' exchanges
        recent_history = history[-2*context_window:] if history else []
        
        # Format as a conversation
        formatted_history = ""
        for message in recent_history:
            role = "user" if message["role"] == "user" else "assistant"
            formatted_history += f"<|begin_of_text|>{role}\n{message['content']}<|end_of_text|>\n"
        
        return formatted_history
    
    def generate_response(self, prompt, session_id, history=None, temperature=0.7, max_tokens=1024, context_window=5):
        """Generate a response from the model given a prompt and conversation history."""
        if not self.llm:
            return {"error": "Model not initialized"}
        
        try:
            # Get or create session history
            if history is not None:
                # Update from client
                self.chat_histories[session_id] = history
            elif session_id not in self.chat_histories:
                self.chat_histories[session_id] = []
            
            # Get current history
            current_history = self.chat_histories[session_id]
            
            # Add current prompt to history only if client didn't provide history
            if history is None:
                current_history.append({"role": "user", "content": prompt})
            
            # Format the prompt with history
            formatted_history = self.format_chat_history(current_history, context_window)
            
            # Prepare sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Format the complete prompt for Llama 3.2
            if formatted_history:
                # Use existing history plus current prompt
                full_prompt = f"{formatted_history}<|begin_of_text|>assistant\n"
            else:
                # First message in conversation
                full_prompt = f"<|begin_of_text|>user\n{prompt}<|end_of_text|>\n<|begin_of_text|>assistant\n"
            
            # Generate response
            outputs = self.llm.generate([full_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # Save the response to history
            current_history.append({"role": "assistant", "content": generated_text})
            self.chat_histories[session_id] = current_history
            
            return {"response": generated_text}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": f"Generation error: {str(e)}"}
    
    def create_new_chat(self, session_id):
        """Create a new chat session."""
        self.chat_histories[session_id] = []
        return {"status": "success", "message": "New chat session created"}
    
    def get_chat_history(self, session_id):
        """Retrieve chat history for a session."""
        if session_id in self.chat_histories:
            return {"history": self.chat_histories[session_id]}
        return {"history": []}

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    history = data.get('history')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 1024)
    context_window = data.get('context_window', 5)
    
    if not prompt:
        return jsonify({"error": "Empty prompt"})
    
    result = app.vllm_app.generate_response(
        prompt=prompt,
        session_id=session_id,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
        context_window=context_window
    )
    return jsonify(result)

@app.route('/new_chat', methods=['POST'])
def new_chat():
    data = request.json
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    result = app.vllm_app.create_new_chat(session_id)
    return jsonify(result)

@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    result = app.vllm_app.get_chat_history(session_id)
    return jsonify(result)

def main():
    parser = argparse.ArgumentParser(description='Run an LLM model with vLLM and a web interface')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the model or Hugging Face model ID')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, 
                        help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9, 
                        help='Fraction of GPU memory to use')
    parser.add_argument('--max-model-len', type=int, default=8192,
                        help='Maximum sequence length the model can handle')
    parser.add_argument('--host', type=str, default='127.0.0.1', 
                        help='Host for the web server')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port for the web server')
    
    args = parser.parse_args()
    
    # Initialize the app
    app.vllm_app = VLLMApp(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # Initialize the model
    if not app.vllm_app.initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        return
    
    # Start the Flask server
    logger.info(f"Starting web server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()