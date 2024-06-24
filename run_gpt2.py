from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Specify the model you want to use (e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
model_name = "gpt2"

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=100, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  # To reduce repetition
        top_k=50,  # To limit the sampling pool
        top_p=0.95,  # To use nucleus sampling
        temperature=0.7,  # To control the randomness
        do_sample=True,  # Enable sampling
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
print("hi, I am your local AI.\n")
prompt = input("How can I help you today?\n")

generated_text = generate_text(prompt)
print(generated_text)

