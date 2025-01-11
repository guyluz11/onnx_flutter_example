import onnxruntime
import numpy as np
from transformers import AutoTokenizer

from helpers.long_text import GETLONGTEXT, SHORTTEXT

# Load the ONNX models for encoder and decoder
encoder_session = onnxruntime.InferenceSession("../models/flassco_encoder_model.onnx")
decoder_session = onnxruntime.InferenceSession("../models/flassco_decoder_model.onnx")

# Initialize tokenizer
# tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")

local_model_dir = "local_model_dir"  # Replace with your local directory path
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)


# Debug special tokens to verify availability
print("CLS Token ID:", tokenizer.cls_token_id)
print("BOS Token ID:", tokenizer.bos_token_id)
print("PAD Token ID:", tokenizer.pad_token_id)
print("EOS Token ID:", tokenizer.eos_token_id)


# Prepare the input text
ARTICLE = GETLONGTEXT
# ARTICLE = "Here is a lot of text I don't want to read ok?"
max_length = 16384

# Tokenize input
inputs = tokenizer(ARTICLE, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

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

# Prepare the decoder input
decoder_input = {
    'encoder_hidden_states': encoder_last_hidden_state.astype(np.float32),  # Ensure correct type
    'encoder_attention_mask': listAttention,  # Add encoder attention mask
    'input_ids': listInputs,  # Add decoder input IDs
}

# Run the decoder
decoder_output = decoder_session.run(None, decoder_input)

print("decoder_output[0]")
print(decoder_output[0])
print()
print("np.argmax(decoder_output[0]")
print(np.argmax(decoder_output[0], axis=-1))

# Extract the final summarized output
summarized_text = tokenizer.decode(np.argmax(decoder_output[0], axis=-1)[0], skip_special_tokens=True)

print("Summarized Text:", summarized_text)
