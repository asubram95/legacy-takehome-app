from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import uvicorn

# Load your trained model once at startup
MODEL_PATH = "../models/tuned_gpt2"
model = None
tokenizer = None

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    max_length: int = 150

class AnswerResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading trained model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")

@app.post("/generate-advice", response_model=AnswerResponse)
async def generate_advice(request: QuestionRequest):
    try:
        # Format input like training data
        input_text = f"Question: {request.question} Answer:"
        
        # Generate response
        ids = tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(
                ids,
                do_sample=True,
                max_length=request.max_length,
                pad_token_id=tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
            )
        
        # Extract answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.replace(input_text, "").strip()
        
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)