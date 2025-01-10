from transformers import AutoModelForSeq2SeqLM

# Load the pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Save the model locally
model.save_pretrained("../assets/bart_model/")
