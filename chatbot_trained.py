import torch
import tiktoken
from src.models.gpt2_model import GPT2Model
from src.models.config import GPT_CONFIG_124M

def load_model(model_path, device):
    """
    Load the model's weights and return the model.
    """
    model = GPT2Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Load only weights
    model.eval()  # Set the model to evaluation mode
    return model

def sample_next_token_nucleus(logits, p=0.9, temperature=1.0):
    """
    Sample the next token using nucleus (top-p) sampling with temperature scaling.
    """
    # Apply temperature scaling to logits
    logits = logits / temperature

    # Clip extreme logits to prevent numerical overflow
    logits = torch.clamp(logits, -10.0, 10.0)

    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=-1)

    # Check for NaN or Inf in probabilities and replace them with zero
    if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
        print("Warning: NaN or Inf found in probs")
        print(f"Logits before softmax: {logits}")
        raise ValueError("NaN or Inf detected in the probability distribution.")

    # Sort probabilities in descending order and get cumulative probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Select only the tokens with cumulative probability less than p
    sorted_indices_to_keep = cumulative_probs < p
    sorted_probs[~sorted_indices_to_keep] = 0  # Set others to zero

    # Add a small epsilon value to avoid division by zero during renormalization
    epsilon = 1e-8
    sorted_probs /= sorted_probs.sum() + epsilon  # Re-normalize to form a valid probability distribution

    # Check for any NaN or Inf values after re-normalization
    if torch.any(torch.isnan(sorted_probs)) or torch.any(torch.isinf(sorted_probs)):
        print("Warning: NaN or Inf found after renormalization")
        print(f"Sorted Probs: {sorted_probs}")
        raise ValueError("NaN or Inf detected after re-normalizing the probability distribution.")

    # Sample the next token from the filtered probabilities
    next_token = torch.multinomial(sorted_probs, 1)

    # Return the corresponding token index from sorted indices
    return sorted_indices.gather(-1, next_token)

def generate_response(model, prompt, tokenizer, max_length=30, device="cpu", temperature=0.7, top_k=20, p=0.9):
    input_tokens = tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)  # Add batch dimension

    generated_tokens = input_tensor

    with torch.no_grad():
        for _ in range(max_length):
            # Get the model's output (logits)
            logits = model(generated_tokens)

            # Focus on the last token's logits
            logits = logits[:, -1, :]

            # Sample the next token using nucleus sampling (top-p)
            next_token = sample_next_token_nucleus(logits, p=p, temperature=temperature)

            # Append the new token to the context
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            # Optionally check for the end-of-text token
            end_of_text_token = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
            if next_token.item() == end_of_text_token:
                break

    # Decode the generated tokens to text
    output_text = tokenizer.decode(generated_tokens.squeeze().cpu().tolist())

    return output_text

def chat(model, tokenizer, device="cpu"):
    """
    Chat interface for interacting with the trained model.
    """
    print("Chatbot ready. Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate response using the trained model
        response = generate_response(model, user_input, tokenizer, device=device, temperature=0.7, top_k=50, p=0.9)
        print(f"Bot: {response}")

def main():
    """
    Main function to initialize and run the chatbot.
    """
    model_path = "gpt2_model.pth"  # Path to the trained model
    
    # Automatically choose device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Load the tokenizer and the trained model
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model(model_path, device)

    # Start the chatbot
    chat(model, tokenizer, device=device)

if __name__ == "__main__":
    main()
