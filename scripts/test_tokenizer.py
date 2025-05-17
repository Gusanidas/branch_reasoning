from transformers import AutoTokenizer

def main():
    # Model name
    model_name = "Qwen/Qwen3-1.7B"
    
    # Download the tokenizer only
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Define chat elements
    system_message = "You are a helpful AI assistant."
    question = "What is the capital of France?"
    answer = "<think> a </think> The capital of France is Paris."
    
    # Format the prompt according to Qwen's chat template
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
        #{"role": "assistant", "content": answer}
    ]
    
    # Use the model's chat template to format the messages
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    # Print the formatted prompt
    print("\nFormatted Prompt:")
    print("----------------")
    print(prompt)
    
    # Also show tokenization details
    tokens = tokenizer.encode(prompt)
    print(f"\nTotal tokens: {len(tokens)}")
    
if __name__ == "__main__":
    main()
