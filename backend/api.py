# api_server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import uvicorn
import logging
import re
import os

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

app = FastAPI(
    title="Mental Health Counseling Assistant API",
    description="AI-powered guidance for mental health professionals",
    version="1.0.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000", 
        "https://legacy-takehome-app.vercel.app/"
        ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models for request/response
class AdviceRequest(BaseModel):
    question: str
    max_length: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

class AdviceResponse(BaseModel):
    answer: str
    success: bool = True
    message: str = "Successfully generated response"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

@app.on_event("startup")
async def load_model():
    """Load the trained model and tokenizer on startup"""
    global model, tokenizer
    
    try:
        model_path = "blobbymagn/legacy-app" # Hugging Face model path
        logger.info(f"Loading model from Hugging Face {model_path}...")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.info("Tokenizer loaded successfully")

        # Load model
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        tokenizer = None

def generate_response(question: str, max_length: int = 200, temperature: float = 0.8, 
                     top_k: int = 50, top_p: float = 0.95) -> str:
    """Generate a response using the trained model"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format input like training data
        input_text = f"Question: {question} Answer:"
        
        # Tokenize input
        device = next(model.parameters()).device
        ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(ids)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                ids,
                do_sample=True,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
                attention_mask=attention_mask,
            )

        # Decode and clean response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.replace(input_text, "").strip()
        logger.info(f"Generated response: {answer[:50]}...")
        cleaned_answer = clean_response(answer)
        logger.info(f"Cleaned response: {cleaned_answer[:50]}...")
        
        return cleaned_answer
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        message="API is running!" if model is not None else "Model not loaded!"
    )

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    # Remove HTML tags and scripts
    text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove dangerous patterns
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    # Escape dangerous characters
    dangerous_chars = {'<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;', '&': '&amp;'}
    for char, escape in dangerous_chars.items():
        text = text.replace(char, escape)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_meaningful_text(text: str) -> bool:
    """Check if text contains meaningful content"""
    
    # Remove whitespace and convert to lowercase
    cleaned = re.sub(r'\s+', '', text.lower())
    
    # Check for repeated characters 
    if re.search(r'(.)\1{3,}', cleaned):  
        return False
    
    # Check for numbers
    if re.search(r'^[\d\s]{3,}$', text.strip()): 
        return False
    
    # Check for minimum word count 
    words = text.strip().split()
    meaningful_words = [word for word in words if len(word) >= 3 and not word.isdigit()]
    
    if len(meaningful_words) < 3:
        return False
    
    return True

def clean_response(text: str, max_words: int = 200) -> str:
    """Clean and limit the model response"""
    
    text = text.strip()

    # Remove everything after "Question:" (case-insensitive)
    question_match = re.search(r'\bQuestion:', text, re.IGNORECASE)
    if question_match:
        text = text[:question_match.start()].strip()
    
    # Split into words and limit
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        text = ' '.join(words)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Use all sentences except the last one
    if len(sentences) > 1:
        complete_text = '. '.join(sentences[:-1]) + '.'
    
    # Remove extra spaces
    complete_text = re.sub(r'\s+', ' ', complete_text)  

    
    return complete_text.strip()

@app.post("/api/generate-advice", response_model=AdviceResponse)
async def generate_advice(request: AdviceRequest):
    """Generate counseling advice"""

    #Sanitize input
    logger.info("Sanitizing input...", sanitize_input(request.question))
    sanitized_question = sanitize_input(request.question)

    # Validate input
    if not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Please enter patient challenge.")
    
    if len(request.question.strip()) < 10:
        raise HTTPException(
            status_code=400, 
            detail="Please provide a more detailed description.")
    
    if len(sanitized_question) > 2000:
        raise HTTPException(
            status_code=400, 
            detail="Your description is too long. Please reduce it to under 2000 characters."
            )
    
    if not is_meaningful_text(sanitized_question):
        raise HTTPException(
            status_code=400, 
            detail="Please provide a more detailed description."
            )
    
    try:
        logger.info(f"Generating response for question: {sanitized_question[:50]}...")
        
        # Generate response
        answer = generate_response(
            question=sanitized_question,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        logger.info("Response generated successfully!")
        
        return AdviceResponse(
            answer=answer,
            success=True,
            message="Response generated successfully!"
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Counseling Assistant API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Configuration
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8000)) 
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )