import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")

# Load the ONNX model
onnx_model_path = "long_t5.onnx"  # Replace with your ONNX model path
sess = ort.InferenceSession(onnx_model_path)

# Input text
long_text = "Here is a lot of text I don't want to read ok?"

max_length = 16384  # Maximum length for the model


# Tokenize the input text
inputs = tokenizer(long_text, return_tensors="np", padding=True, truncation=True, max_length=max_length)

# Prepare inputs for the ONNX model
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Convert inputs to the correct format (ONNX Runtime expects numpy arrays)
input_ids = np.array(input_ids, dtype=np.int64)
attention_mask = np.array(attention_mask, dtype=np.int64)


# Prepare decoder_input_ids (usually, we use the <BOS> token as the starting token for the decoder)
decoder_input_ids = np.full_like(input_ids, tokenizer.pad_token_id)  # Initialize with padding token ID

# You can also initialize decoder_input_ids with the <BOS> token (if available)
# decoder_input_ids[0] = tokenizer.convert_tokens_to_ids("<BOS>")  # Uncomment if <BOS> is available


# Prepare inputs dictionary for ONNX model
onnx_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "decoder_input_ids": decoder_input_ids
}

# Run the model and get the output
outputs = sess.run(None, onnx_inputs)

# The output is usually a list of logits from the model
logits = outputs[0]

# Convert logits to text
# For simplicity, you can use the tokenizer's `decode` method to extract the summary from the logits
summary_ids = np.argmax(logits, axis=-1)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print(summary)
