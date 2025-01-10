

import onnxruntime
import numpy as np
from transformers import AutoTokenizer

# Load the ONNX models for encoder and decoder
encoder_session = onnxruntime.InferenceSession("flassco_encoder_model.onnx")
decoder_session = onnxruntime.InferenceSession("flassco_decoder_model.onnx")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")

# Prepare the input text
ARTICLE = """Your article content here."""

# Tokenize input
inputs = tokenizer(ARTICLE, return_tensors="pt", padding=True, truncation=True)

listInputs =  inputs['input_ids'].cpu().numpy()
listAttention = inputs['attention_mask'].cpu().numpy()

# Generate encoder input (use attention_mask in addition to input_ids)
encoder_inputs = {
    'input_ids': listInputs,
    'attention_mask': listAttention,
}

print(encoder_session.get_inputs())
# Run the encoder
encoder_output = encoder_session.run(None, encoder_inputs)

# Extract the encoder outputs (this depends on your model, adjust if needed)
encoder_last_hidden_state = encoder_output[0]  # Assuming the first output is hidden states

# Prepare the decoder input (adjust based on your decoder's needs)
decoder_input = {
    'input_ids': encoder_last_hidden_state,
    # Optionally, add additional inputs here
}

# Run the decoder
decoder_output = decoder_session.run(None, decoder_input)

# Extract the final summarized output (this will depend on the model, adjust accordingly)
summarized_text = tokenizer.decode(np.argmax(decoder_output[0], axis=-1)[0], skip_special_tokens=True)

print(summarized_text)
