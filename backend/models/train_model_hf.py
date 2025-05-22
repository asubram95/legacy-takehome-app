# train_gpt2_mental_health.py

from transformers import (
   TextDataset, 
   DataCollatorForLanguageModeling,
   GPT2Tokenizer,
   GPT2LMHeadModel,
   Trainer, 
   TrainingArguments
)
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Helper Functions
def load_dataset(file_path, tokenizer, block_size=512):
   dataset_train = TextDataset(
       tokenizer=tokenizer,
       file_path=file_path,
       block_size=block_size,
   )
   return dataset_train

def load_data_collator(tokenizer, mlm=False):
   data_collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer, 
       mlm=mlm,
   )
   return data_collator

def train_model(train_file_path, model_name, output_dir, overwrite_output_dir,
               per_device_train_batch_size, num_train_epochs):
   
   print("GPT and tokenizer loaded!")
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)
   tokenizer.pad_token = tokenizer.eos_token
   
   # Load datasets
   print("\nLoading training dataset...")
   train_dataset = load_dataset(train_file_path, tokenizer)
   print(f"Training dataset size: {len(train_dataset)}")

   # Load data collator
   data_collator = load_data_collator(tokenizer)

   # Save tokenizer
   tokenizer.save_pretrained(output_dir)

   # Load model
   model = GPT2LMHeadModel.from_pretrained(model_name)
   
   # Display model architecture
   print("Model architecture:")
   print(model.transformer)

   # Save initial model
   model.save_pretrained(output_dir)

   # Training arguments
   training_args = TrainingArguments(
       output_dir=output_dir,
       overwrite_output_dir=overwrite_output_dir,
       num_train_epochs=num_train_epochs,
       per_device_train_batch_size=per_device_train_batch_size,
       logging_dir="./logs",
       logging_steps=100,
       save_steps=50,
       logging_first_step=True,
       save_total_limit=2,
       learning_rate=0.0001,
       warmup_steps=50,
       dataloader_num_workers=4,
       no_cuda=False, 
       dataloader_pin_memory=True,
       fp16=True,
   )
   
   trainer = Trainer(
       model=model,
       args=training_args,
       data_collator=data_collator,
       train_dataset=train_dataset,
   )
   
   # Train model
   print("\nTraining...")
   hist = trainer.train()
   
   # Save final model
   trainer.save_model()
   
   return trainer, hist

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
   # Read Data
   print("Reading data...")
   data = pd.read_csv('../data/processed.csv', usecols=['context', 'response'])
   print(f"Data shape: {data.shape}")
   print("Sample data:")
   print(data.sample(5))
   print(data.info())
   print()

   #Prep Data
   print("Preparing data...")

   # Create training text file for GPT-2 
   train_file_path = "../data/train_text.txt"
   with open(train_file_path, 'w', encoding='utf-8') as f:
       for _, row in data.iterrows():
           # Format: Question: {context} Answer: {response}
           text = f"Question: {row['context']} Answer: {row['response']}\n"
           f.write(text)

   print(f"Created training file: {train_file_path}")

   #Training Configuration
   print("\nConfiguring training...")

   model_name = 'gpt2'
   output_dir = '../models/tuned_gpt2'
   overwrite_output_dir = True
   batch_size = 8
   epochs = 3

   print(f"Model: {model_name}")
   print(f"Output directory: {output_dir}")
   print(f"Batch size: {batch_size}")
   print(f"Epochs: {epochs}")
   print()

   #Training
   print("Loading model...")

   model = train_model(
       train_file_path=train_file_path,
       model_name=model_name,
       output_dir=output_dir,
       overwrite_output_dir=overwrite_output_dir,
       per_device_train_batch_size=batch_size,
       num_train_epochs=epochs,
   )

   print("Training completed!")
   print()

   # Evaluation
   print("Training Results...")

   print(f"Global Step: {model.global_step}")
   print(f"Epoch: {model.metrics['epoch']}")
   print(f"Train Runtime: {model.metrics['train_runtime']:.2f} seconds")
   print(f"Train Samples Per Second: {model.metrics['train_samples_per_second']:.3f}")
   print(f"Train Steps Per Second: {model.metrics['train_steps_per_second']:.3f}")
   print(f"Train Loss: {model.metrics['train_loss']:.6f}")
   print()

   # Generation Functions
   print("Generation Functions...")

   #Test Generation
   print("Testing Generation...")

   # Load the trained model
   model = load_trained_model(output_dir)
   tokenizer = load_trained_tokenizer(output_dir)

   # Test with some examples from the dataset
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

if __name__ == "__main__":
   main()