
from transformers import AutoTokenizer

# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Save it locally
tokenizer.save_pretrained("../assets/bart_tokenizer/")
