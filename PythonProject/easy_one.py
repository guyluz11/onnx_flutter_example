from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(
    "pszemraj/bigbird-pegasus-large-K-booksum",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "pszemraj/bigbird-pegasus-large-K-booksum",
)

# Create the summarizer pipeline
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
)

# The input text
wall_of_text = "your"

# Tokenization: see what the tokenizer produces
tokens = tokenizer.tokenize(wall_of_text)
print("Tokens:", tokens)

# Optionally, you can also check the token IDs (numeric representation)
token_ids = tokenizer.encode(wall_of_text)
print("Token IDs:", token_ids)

# Summarize the text
result = summarizer(
    wall_of_text,
    min_length=16,
    max_length=256,
    no_repeat_ngram_size=3,
    clean_up_tokenization_spaces=True,
)

# Output the summary
print("Summary:", result[0]["summary_text"])
