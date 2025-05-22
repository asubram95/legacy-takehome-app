from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import random

def load_trained_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_trained_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_response(model, tokenizer, question, max_length=150):
    # Format the input like training data
    input_text = f"Question: {question} Answer:"
    
    ids = tokenizer.encode(input_text, return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
    )
    
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    answer = generated_text.replace(input_text, "").strip()
    return answer

def main():
    # Configuration
    model_path = './tuned_gpt2'
    data_path = '../data/processed.csv'
    
    print("Loading model...")
    
    # Load trained model
    model = load_trained_model(model_path)
    tokenizer = load_trained_tokenizer(model_path)
    
    # Load data 
    data = pd.read_csv(data_path, usecols=['context', 'response'])
    
    # Test with some examples from the dataset
    random_indices = random.sample(range(len(data)), 3)

    test_questions = [
        data['context'].iloc[random_indices[0]],
        data['context'].iloc[random_indices[1]],
        data['context'].iloc[random_indices[2]],
        "How can I help a patient who is feeling anxious?"
    ] 

    print("Testing model...")

    for i, question in enumerate(test_questions):
        print(f"Q{i+1}: {question}")
        
        if i < 3:
            print(f"\nOriginal: {data['response'].iloc[random_indices[i]]}")
        
        generated_answer = generate_response(model, tokenizer, question)
        print(f"\nGenerated: {generated_answer}")
        print()

    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main()

