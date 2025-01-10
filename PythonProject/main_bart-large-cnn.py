

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from long_text import LONGTEXT

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Tokenize the input text
inputs = tokenizer(LONGTEXT, return_tensors="pt")

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
print(inputs)
print()
print(inputs["input_ids"])
print()
# Generate summary
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)



#
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# print(model.summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
# # >>> [{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]





