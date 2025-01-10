


import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from long_text import LONGTEXT
#

import onnx

# Load the model
model_path = "assets/model/model.onnx"
model = onnx.load(model_path)

# Check if the model is valid
onnx.checker.check_model(model)
print("The model is valid.")




# Load ONNX model using onnxruntime
onnx_model_path = "assets/model/model.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# Load tokenizer (can be local as mentioned before)
tokenizer = AutoTokenizer.from_pretrained("assets/bart_tokenizer/")

# Tokenize the input text
inputs = tokenizer(LONGTEXT, return_tensors="np", padding=True, truncation=True)

# Convert inputs to NumPy array for ONNX compatibility
input_ids = np.array(inputs["input_ids"])

# Perform inference with ONNX model
inputs_onnx = {session.get_inputs()[0].name: input_ids}  # Get input node name dynamically
output = session.run(None, inputs_onnx)

# Assuming the model output is text generation ids, decode them
summary = tokenizer.decode(output[0][0], skip_special_tokens=True)

# Print the summary
print(summary)
