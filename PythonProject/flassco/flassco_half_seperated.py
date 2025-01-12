import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from helpers.long_text import GETLONGTEXT

# Define the Hugging Face model name
model_name = "Falconsai/text_summarization"


local_model_dir = "local_model_dir"  # Replace with your local directory path
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)


# Load the model (encoder and decoder) using Hugging Face
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare the input text

# Tokenize the input text using Hugging Face tokenizer
inputs = tokenizer(GETLONGTEXT, return_tensors="pt", padding=True, truncation=True)

# Run the model to generate the summary
summary_ids = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=150,  # Adjust summary length as needed
    num_beams=4,     # Use beam search for better results
    early_stopping=True
)

# Decode the summarized text using Hugging Face tokenizer
summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Summarized Text:", summarized_text)
