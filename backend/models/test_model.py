from transformers import (
   GPT2Tokenizer,
   GPT2LMHeadModel,
)
import pandas as pd

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

def test_model():
# Evaluation
   print("Training Results...")

   print(f"Global Step: {model[0].global_step}")
   print(f"Epoch: {model[0].metrics['epoch']}")
   print(f"Train Runtime: {model[0].metrics['train_runtime']:.2f} seconds")
   print(f"Train Samples Per Second: {model[0].metrics['train_samples_per_second']:.3f}")
   print(f"Train Steps Per Second: {model[0].metrics['train_steps_per_second']:.3f}")
   print(f"Train Loss: {model[0].metrics['train_loss']:.6f}")
   print()

   # Generation Functions
   print("Generation Functions...")

   #Test Generation
   print("Testing Generation...")

   # Load the trained model
   output_dir = '../models/tuned_gpt2'
   model = load_trained_model(output_dir)
   tokenizer = load_trained_tokenizer(output_dir)

   # Test with some examples from the dataset
   data = pd.read_csv('../data/processed.csv', usecols=['context', 'response'])
   test_questions = [
       data['context'].iloc[0],
       data['context'].iloc[10],
       "How can I help a patient who is feeling anxious?"
   ]

   print("Testing generation with sample questions:")

   for i, question in enumerate(test_questions):
       print(f"\nTest {i+1}:")
       print(f"Q: {question}")
       
       if i < 2:  # Show original answer for first two
           print(f"Original A: {data['response'].iloc[i*10]}")
       
       generated_answer = generate_response(model, tokenizer, question)
       print(f"Generated A: {generated_answer}")
       print("-" * 40)

   print("\nTraining and testing completed successfully!")

def main():
    test_model()

if __name__ == "__main__":
   main()