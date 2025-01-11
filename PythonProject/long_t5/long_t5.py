import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from helpers.long_text import GETLONGTEXT

# https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary

# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")

# Load the tokenizer from a local directory
local_model_dir = "local_model_dir"  # Replace with your local directory path
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# Test the tokenizer
print('Go')
print(tokenizer.decode([
        947,
        19,
        3,
        9,
        418,
        13,
        1499,
        27,
        278,
        31,
        17,
        241,
        12,
        608,
        3,
        1825,
        58,
        1,], skip_special_tokens=True))
print('Go')

#
#
# # Load the ONNX model
onnx_model_path = "../models/long_t5.onnx"  # Replace with your ONNX model path
sess = ort.InferenceSession(onnx_model_path)
#
#
# # Prepare the input text

max_length = 16384  # Maximum length for the model
output_max_length = 512  # Set a reasonable length for the summary
#
# # Tokenize the input text
inputs = tokenizer(GETLONGTEXT, return_tensors="np", padding=True, truncation=True, max_length=max_length)
# # [566, 15, 40, 40, 32]
# # Prepare inputs for the ONNX model
#
input_ids = np.array(inputs["input_ids"], dtype=np.int64)
print(input_ids )
print(inputs["input_ids"])
# print('input_ids')
# print(tokenizer.decode([
#     19,
#   ], skip_special_tokens=True))
attention_mask = np.array(inputs["attention_mask"], dtype=np.int64)
#
# # Initialize the decoder with a starting token (fallback to the pad token if bos_token_id is None)
start_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id
decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)  # Begin with the start token
#
# Iteratively decode the output
generated_tokens = []

for _ in range(output_max_length):
    # Prepare inputs dictionary for ONNX model
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids
    }

    # Run the model
    outputs = sess.run(None, onnx_inputs)

    # Extract logits and get the most probable next token
    logits = outputs[0]



    next_token_id = np.argmax(logits[0, -1])  # Get the last token's prediction

    # Add the new token to the generated sequence
    generated_tokens.append(next_token_id)
    # Break if the model predicts the end-of-sequence token
    if next_token_id == tokenizer.eos_token_id:
        break

    # Update decoder input for the next step
    decoder_input_ids = np.concatenate([decoder_input_ids, [[next_token_id]]], axis=1)

print('decoder_input_ids')
print(decoder_input_ids)
# Decode the generated token IDs into text
summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Print the summary
print(summary)
