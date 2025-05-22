from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import os
import json
from pydantic import BaseModel
from typing import List, Optional
import openai
import pickle
from contextlib import contextmanager

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Counselor API",
    description="API for mental health counselors to get guidance on patient care",
    version="1.0.0"
)

# Set up CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_PATH = "data/processed/conversations.db"
MODEL_PATH = "models/response_type_model.pkl"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pydantic Models for Request/Response
class Conversation(BaseModel):
    id: str
    patient_message: str
    therapist_response: str
    response_type: str
    patient_msg_length: Optional[int] = None
    therapist_resp_length: Optional[int] = None

class SearchResponse(BaseModel):
    conversations: List[Conversation]
    total: int
    limit: int
    offset: int

class PredictionRequest(BaseModel):
    patient_message: str

class PredictionResponse(BaseModel):
    response_type: str
    confidence: float
    similar_cases: List[Conversation]

class LLMRequest(BaseModel):
    counselor_challenge: str

class LLMResponse(BaseModel):
    suggestion: str
    relevant_cases: Optional[List[Conversation]] = []

class StatsResponse(BaseModel):
    total_conversations: int
    response_type_distribution: dict
    avg_patient_msg_length: float
    avg_therapist_resp_length: float

# Database connection helper
@contextmanager
def get_db_connection():
    """Create a connection to the SQLite database with context management"""
    if not os.path.exists(DATABASE_PATH):
        raise HTTPException(status_code=500, detail="Database not found. Please run the data pipeline first.")
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Model loading helper
def load_model():
    """Load the trained ML model"""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
    return None

# Simple rule-based fallback classifier
def simple_classify(text: str) -> tuple[str, float]:
    """Simple rule-based classification as fallback"""
    text = text.lower()
    if any(word in text for word in ['suggest', 'recommend', 'try', 'should', 'could']):
        return 'direct_advice', 0.7
    elif any(word in text for word in ['feel', 'seem', 'sounds like', 'appears']):
        return 'reflection', 0.8
    elif any(word in text for word in ['?', 'tell me more', 'could you', 'what', 'how', 'why']):
        return 'question', 0.6
    else:
        return 'other', 0.5

# Generate rule-based suggestions
def generate_suggestion(challenge: str, context: str) -> str:
    """Generate a simple rule-based suggestion"""
    challenge_lower = challenge.lower()
    
    suggestions = []
    
    # Anxiety-related suggestions
    if any(word in challenge_lower for word in ['anxious', 'anxiety', 'worried', 'panic']):
        suggestions.extend([
            "Consider using grounding techniques like the 5-4-3-2-1 method (5 things you see, 4 you hear, etc.).",
            "Explore breathing exercises such as box breathing or progressive muscle relaxation.",
            "Help the patient identify specific anxiety triggers and develop coping strategies."
        ])
    
    # Depression-related suggestions
    if any(word in challenge_lower for word in ['depressed', 'depression', 'sad', 'hopeless']):
        suggestions.extend([
            "Focus on small, achievable daily goals to build momentum and self-efficacy.",
            "Consider cognitive behavioral therapy techniques to address negative thought patterns.",
            "Explore the patient's support system and ways to strengthen social connections."
        ])
    
    # Anger-related suggestions
    if any(word in challenge_lower for word in ['angry', 'anger', 'frustrated', 'irritated']):
        suggestions.extend([
            "Help the patient identify triggers and early warning signs of anger.",
            "Teach anger management techniques like counting to ten or taking a timeout.",
            "Explore underlying emotions that might be masked by anger (hurt, fear, etc.)."
        ])
    
    # Trauma-related suggestions
    if any(word in challenge_lower for word in ['trauma', 'ptsd', 'flashback', 'nightmare']):
        suggestions.extend([
            "Ensure the patient feels safe and supported in the therapeutic environment.",
            "Consider trauma-informed care approaches and grounding techniques.",
            "Be mindful of potential triggers and move at the patient's pace."
        ])
    
    # Relationship issues
    if any(word in challenge_lower for word in ['relationship', 'partner', 'family', 'conflict']):
        suggestions.extend([
            "Explore communication patterns and help develop healthy communication skills.",
            "Consider family or couples therapy if appropriate.",
            "Help the patient set healthy boundaries in relationships."
        ])
    
    # General suggestions if no specific category matches
    if not suggestions:
        suggestions.extend([
            "Start with active listening and validation of the patient's feelings.",
            "Ask open-ended questions to better understand their perspective and experiences.",
            "Use reflective listening to ensure the patient feels heard and understood.",
            "Consider the patient's cultural background and how it might influence their experience."
        ])
    
    return " ".join(suggestions[:3])  # Limit to first 3 suggestions to keep response manageable

# API Routes
@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint"""
    return {
        "message": "Mental Health Counselor API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/conversations", response_model=SearchResponse, tags=["Conversations"])
def get_conversations(
    query: Optional[str] = Query(None, description="Search query for patient messages or therapist responses"),
    response_type: Optional[str] = Query(None, description="Filter by response type"),
    limit: int = Query(20, ge=1, le=100, description="Number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip")
):
    """Search for conversations based on query and filters"""
    try:
        with get_db_connection() as conn:
            # Base query
            sql_query = "SELECT * FROM conversations"
            conditions = []
            params = []
            
            # Add search query if provided
            if query:
                conditions.append("(patient_message LIKE ? OR therapist_response LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%"])
            
            # Add response type filter if provided
            if response_type:
                conditions.append("response_type = ?")
                params.append(response_type)
            
            # Complete the query
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            
            # Add pagination
            sql_query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            cursor = conn.execute(sql_query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            # Get total count for pagination
            count_query = "SELECT COUNT(*) as total FROM conversations"
            if conditions:
                count_query += " WHERE " + " AND ".join(conditions)
            total = conn.execute(count_query, params[:-2] if params else []).fetchone()["total"]
            
            return SearchResponse(
                conversations=results,
                total=total,
                limit=limit,
                offset=offset
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
def predict_response_type(request: PredictionRequest):
    """Predict the response type for a patient message"""
    try:
        patient_message = request.patient_message.strip()
        
        if not patient_message:
            raise HTTPException(status_code=400, detail="patient_message cannot be empty")
        
        # Load the model
        model = load_model()
        
        if model is None:
            # Fallback to simple rule-based classification
            predicted_type, confidence = simple_classify(patient_message)
        else:
            # Use the trained model
            try:
                prediction = model.predict([patient_message])[0]
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([patient_message])[0]
                    confidence = max(proba)
                else:
                    confidence = 0.8  # Default confidence
                
                predicted_type = prediction
            except Exception as e:
                print(f"Model prediction failed: {e}, falling back to rule-based")
                predicted_type, confidence = simple_classify(patient_message)
        
        # Find similar cases
        with get_db_connection() as conn:
            # Simple similarity search using keywords from the patient message
            words = [word for word in patient_message.split()[:5] if len(word) > 3]
            
            if words:
                search_conditions = " OR ".join([f"patient_message LIKE '%{word}%'" for word in words])
                cursor = conn.execute(f"SELECT * FROM conversations WHERE {search_conditions} LIMIT 5")
                similar_cases = [dict(row) for row in cursor.fetchall()]
            else:
                similar_cases = []
        
        return PredictionResponse(
            response_type=predicted_type,
            confidence=float(confidence),
            similar_cases=similar_cases
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/suggest", response_model=LLMResponse, tags=["AI Suggestions"])
def get_suggestion(request: LLMRequest):
    """Get a suggestion for the counselor based on their challenge"""
    try:
        counselor_challenge = request.counselor_challenge.strip()
        
        if not counselor_challenge:
            raise HTTPException(status_code=400, detail="counselor_challenge cannot be empty")
        
        # Find relevant cases from our database
        with get_db_connection() as conn:
            words = [word for word in counselor_challenge.split()[:5] if len(word) > 3]
            
            if words:
                search_conditions = " OR ".join([f"patient_message LIKE '%{word}%'" for word in words])
                cursor = conn.execute(f"SELECT * FROM conversations WHERE {search_conditions} LIMIT 3")
                relevant_cases = [dict(row) for row in cursor.fetchall()]
            else:
                relevant_cases = []
        
        # Create context for suggestions
        context = "\n\n".join([
            f"Patient: {case['patient_message']}\nTherapist: {case['therapist_response']}"
            for case in relevant_cases
        ]) if relevant_cases else "No directly relevant cases found."
        
        # Generate suggestion (using rule-based approach for POC)
        suggestion = generate_suggestion(counselor_challenge, context)
        
        # If you have OpenAI API key, you could enhance this:
        if OPENAI_API_KEY and len(counselor_challenge) > 10:
            try:
                # This is where you'd make an OpenAI API call
                # For now, we'll stick with the rule-based approach
                pass
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
        
        return LLMResponse(
            suggestion=suggestion,
            relevant_cases=relevant_cases
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion error: {str(e)}")

@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
def get_stats():
    """Get database statistics and analytics"""
    try:
        with get_db_connection() as conn:
            # Get total conversations
            total_result = conn.execute("SELECT COUNT(*) as count FROM conversations").fetchone()
            total_conversations = total_result["count"]
            
            # Get response type distribution
            response_types = conn.execute("""
                SELECT response_type, COUNT(*) as count 
                FROM conversations 
                GROUP BY response_type
                ORDER BY count DESC
            """).fetchall()
            
            # Get average message lengths
            avg_lengths = conn.execute("""
                SELECT 
                    AVG(LENGTH(patient_message)) as avg_patient_length,
                    AVG(LENGTH(therapist_response)) as avg_therapist_length
                FROM conversations
            """).fetchone()
            
            return StatsResponse(
                total_conversations=total_conversations,
                response_type_distribution={row["response_type"]: row["count"] for row in response_types},
                avg_patient_msg_length=float(avg_lengths["avg_patient_length"] or 0),
                avg_therapist_resp_length=float(avg_lengths["avg_therapist_length"] or 0)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.get("/response-types", tags=["Analytics"])
def get_response_types():
    """Get all available response types"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT response_type FROM conversations ORDER BY response_type")
            response_types = [row["response_type"] for row in cursor.fetchall()]
            
            return {"response_types": response_types}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching response types: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)