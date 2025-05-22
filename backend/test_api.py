import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_search_conversations():
    """Test searching conversations"""
    print("🔍 Testing conversation search...")
    
    # Test basic search
    response = requests.get(f"{BASE_URL}/conversations", params={"limit": 5})
    print(f"Basic search - Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['total']} total conversations, showing {len(data['conversations'])}")
        if data['conversations']:
            print(f"First conversation: {data['conversations'][0]['patient_message'][:100]}...")
    
    # Test search with query
    response = requests.get(f"{BASE_URL}/conversations", params={"query": "anxiety", "limit": 3})
    print(f"Search with 'anxiety' - Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['total']} conversations with 'anxiety'")
    
    print("-" * 50)

def test_prediction():
    """Test response type prediction"""
    print("🔍 Testing response type prediction...")
    
    test_messages = [
        "I feel very anxious about my upcoming presentation at work",
        "I can't stop thinking about what happened last week",
        "My partner and I keep fighting about the same things"
    ]
    
    for message in test_messages:
        payload = {"patient_message": message}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Message: {message[:50]}...")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Predicted type: {data['response_type']} (confidence: {data['confidence']:.2f})")
            print(f"Similar cases found: {len(data['similar_cases'])}")
        print()
    
    print("-" * 50)

def test_suggestions():
    """Test AI suggestions"""
    print("🔍 Testing AI suggestions...")
    
    test_challenges = [
        "My patient is dealing with severe anxiety and panic attacks",
        "I have a client who seems depressed but won't open up",
        "My patient has anger management issues and gets frustrated easily"
    ]
    
    for challenge in test_challenges:
        payload = {"counselor_challenge": challenge}
        response = requests.post(f"{BASE_URL}/suggest", json=payload)
        print(f"Challenge: {challenge[:50]}...")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Suggestion: {data['suggestion'][:150]}...")
            print(f"Relevant cases: {len(data['relevant_cases'])}")
        print()
    
    print("-" * 50)

def test_stats():
    """Test statistics endpoint"""
    print("🔍 Testing statistics...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total conversations: {data['total_conversations']}")
        print(f"Response types: {data['response_type_distribution']}")
        print(f"Avg patient message length: {data['avg_patient_msg_length']:.1f}")
        print(f"Avg therapist response length: {data['avg_therapist_resp_length']:.1f}")
    
    print("-" * 50)

def test_response_types():
    """Test response types endpoint"""
    print("🔍 Testing response types...")
    response = requests.get(f"{BASE_URL}/response-types")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Available response types: {data['response_types']}")
    
    print("-" * 50)

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting FastAPI tests...\n")
    
    try:
        test_health_check()
        test_search_conversations()
        test_prediction()
        test_suggestions()
        test_stats()
        test_response_types()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error! Make sure the FastAPI server is running on http://localhost:8000")
        print("   Run: uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    run_all_tests()