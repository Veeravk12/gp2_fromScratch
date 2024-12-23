import torch
import tiktoken  # Assuming tiktoken is used for GPT-2 tokenizer
from src.models.gpt2_model import GPT2Model  # Assuming GPT2Model is imported correctly
from src.models.config import GPT_CONFIG_124M  # Import the GPT config (adjust as necessary)
import os

# Function to decode token IDs to text
def decode(tokens, tokenizer):
    """
    Converts a list of token IDs back to human-readable text using the tokenizer.
    
    Args:
    - tokens (list): List of token IDs.
    - tokenizer (tiktoken.Encoding): The tokenizer used for encoding/decoding.
    
    Returns:
    - str: The decoded text.
    """
    decoded_text = tokenizer.decode(tokens)
    
    # Optionally remove the special token <|endoftext|> from the decoded text
    # This is a common GPT-2 special token that is used to mark the end of a text
    decoded_text = decoded_text.replace("<|endoftext|>", "").strip()

    return decoded_text

# Function to generate text based on the input
def generate_response(model, input_tokens, max_new_tokens, context_size):
    """
    Generates text by predicting the next token iteratively using the given model.

    Args:
    - model (nn.Module): The GPT-2 model.
    - input_tokens (torch.Tensor): A tensor of token IDs for the conversation.
    - max_new_tokens (int): The number of new tokens to generate.
    - context_size (int): The maximum length of the context (max number of tokens the model can handle).

    Returns:
    - torch.Tensor: The tensor containing the generated sequence of tokens.
    """
    # Ensure the input doesn't exceed the max context size
    input_tokens = input_tokens[:, -context_size:]

    # Generate the response
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tokens)

        logits = logits[:, -1, :]  # Focus on the last token's logits
        probas = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        next_token = torch.argmax(probas, dim=-1, keepdim=True)  # Pick the most probable token

        input_tokens = torch.cat((input_tokens, next_token), dim=1)  # Append the new token to the context

    return input_tokens

def main():
    """
    Main function to run inference and simulate conversation-like responses.
    """
    # Initialize device (cuda or cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained GPT-2 model
    model = GPT2Model().to(device)
    model.eval()  # Set the model to evaluation mode

    # Load the tokenizer (make sure it's the one from tiktoken for GPT-2)
    tokenizer = tiktoken.get_encoding("gpt2")  # Use the GPT-2 tokenizer

    # Set the maximum length of the context the model can handle
    context_size = GPT_CONFIG_124M["context_length"]  # Adjust according to your model config

    # Set the maximum number of new tokens to generate per response
    max_new_tokens = 50  # You can adjust this based on your preference

    # Initialize conversation history as empty list
    conversation_history = []

    # Start the chat loop
    print("My Bot: Hi! How can I assist you today?")
    
    while True:
        # Ask the user for input
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:  # End the conversation
            print("My Bot: Goodbye! Have a great day!")
            break

        # Append user input to the conversation history
        conversation_history.append(f"You: {user_input}")

        # Join the conversation history and tokenize it
        full_conversation = " ".join(conversation_history)
        
        # Use tiktoken to encode the full conversation
        input_tokens = tokenizer.encode(full_conversation)
        
        # Convert the list of tokens into a PyTorch tensor
        input_tokens = torch.tensor([input_tokens]).to(device)

        # Generate a response
        generated_tokens = generate_response(model, input_tokens, max_new_tokens, context_size)

        # Decode the generated tokens to text
        generated_text = decode(generated_tokens[0].cpu().numpy().tolist(), tokenizer)

        # Extract the model's response (last sentence) from the generated text
        model_response = generated_text[len(full_conversation):].strip()

        # Append the model's response to the conversation history
        conversation_history.append(f"My Bot: {model_response}")

        # Print the model's response
        print(f"My Bot: {model_response}")

if __name__ == "__main__":
    # Run the main function
    main()
