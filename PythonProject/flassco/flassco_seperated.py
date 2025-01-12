import re
import onnxruntime
import numpy as np
from transformers import AutoTokenizer

from helpers.long_text import GETLONGTEXT

# Function to preprocess the input text
def preprocess_text(text):
    # Remove consecutive punctuation (e.g., `...,`)
    text = re.sub(r'[,.]{2,}', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Lowercase the text (optional)
    text = text.lower()
    return text

# Load the ONNX models for encoder and decoder
encoder_session = onnxruntime.InferenceSession("../models/flassco_encoder_model.onnx")
decoder_session = onnxruntime.InferenceSession("../models/flassco_decoder_model.onnx")

# Initialize tokenizer
local_model_dir = "local_model_dir"  # Replace with your local directory path
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# Debug special tokens to verify availability
print("CLS Token ID:", tokenizer.cls_token_id)
print("BOS Token ID:", tokenizer.bos_token_id)
print("PAD Token ID:", tokenizer.pad_token_id)
print("EOS Token ID:", tokenizer.eos_token_id)

# Prepare the input text
ARTICLE = preprocess_text(GETLONGTEXT)  # Preprocess the text
# ARTICLE = GETLONGTEXT  # Preprocess the text
max_summary_length = 100  # Maximum number of tokens for the summary

# Tokenize input
inputs = tokenizer(ARTICLE, return_tensors="pt", padding=True, truncation=True)

listInputs = inputs['input_ids'].cpu().numpy()
listAttention = inputs['attention_mask'].cpu().numpy()

# Generate encoder input (use attention_mask in addition to input_ids)
encoder_inputs = {
    'input_ids': listInputs,
    'attention_mask': listAttention,
}

# Run the encoder
encoder_output = encoder_session.run(None, encoder_inputs)

# Extract the encoder outputs
encoder_last_hidden_state = encoder_output[0]  # Assuming the first output is hidden states

print('encoder_last_hidden_state')
print(encoder_last_hidden_state)

# Check and set the starting token ID
start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id or 0

# Prepare initial decoder input IDs
initial_decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)


# Prepare the decoder input with maximum summary length
decoder_input = {
    'encoder_hidden_states': encoder_last_hidden_state.astype(np.float32),  # Ensure correct type
    'encoder_attention_mask': listAttention,  # Add encoder attention mask
    'decoder_input_ids': initial_decoder_input_ids,
}

# Run the decoder iteratively to respect the maximum summary length
current_output = initial_decoder_input_ids
for _ in range(max_summary_length):
    decoder_output = decoder_session.run(None, {
        'encoder_hidden_states': encoder_last_hidden_state.astype(np.float32),
        'encoder_attention_mask': listAttention,
        'input_ids': current_output,
    })

    # Get the next token (argmax of logits)
    next_token_id = np.argmax(decoder_output[0][:, -1, :], axis=-1).reshape(-1, 1)

    # Append the new token to the current output
    current_output = np.hstack([current_output, next_token_id])

    # Stop if the [EOS] token is generated
    if next_token_id[0, 0] == tokenizer.eos_token_id:
        break

# Decode the summarized text
summarized_text = tokenizer.decode(current_output[0])

print("Summarized Text:", summarized_text)
