import numpy as np
from transformers import AutoTokenizer


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
# onnx_model_path = "models/long_t5.onnx"  # Replace with your ONNX model path
# sess = ort.InferenceSession(onnx_model_path)
#
#
# # Prepare the input text
GETLONGTEXT = "Here is a lot of text I don't want to read ok?"
# # """New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
# # A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
# # Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
# # In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
# # Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
# # 2010 marriage license application, according to court documents.
# # Prosecutors said the marriages were part of an immigration scam.
# # On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
# # After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
# # Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
# # All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
# # Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
# # Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
# # The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
# # Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
# # Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
# # If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
# # """
max_length = 16384  # Maximum length for the model
# output_max_length = 512  # Set a reasonable length for the summary
#
# # Tokenize the input text
inputs = tokenizer(GETLONGTEXT, return_tensors="np", padding=True, truncation=True, max_length=max_length)
# # [566, 15, 40, 40, 32]
# # Prepare inputs for the ONNX model
#
input_ids = np.array(inputs["input_ids"], dtype=np.int64)
print(input_ids )
print()
# print('input_ids')
# print(tokenizer.decode([
#     19,
#   ], skip_special_tokens=True))
# attention_mask = np.array(inputs["attention_mask"], dtype=np.int64)
#
# # Initialize the decoder with a starting token (fallback to the pad token if bos_token_id is None)
# start_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id
# decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)  # Begin with the start token
#
# # Iteratively decode the output
# generated_tokens = []
#
# for _ in range(output_max_length):
#     # Prepare inputs dictionary for ONNX model
#     onnx_inputs = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "decoder_input_ids": decoder_input_ids
#     }
#
#     # Run the model
#     outputs = sess.run(None, onnx_inputs)
#
#     # Extract logits and get the most probable next token
#     logits = outputs[0]
#
#
#
#     next_token_id = np.argmax(logits[0, -1])  # Get the last token's prediction
#
#     # Add the new token to the generated sequence
#     generated_tokens.append(next_token_id)
#     # Break if the model predicts the end-of-sequence token
#     if next_token_id == tokenizer.eos_token_id:
#         break
#
#     # Update decoder input for the next step
#     decoder_input_ids = np.concatenate([decoder_input_ids, [[next_token_id]]], axis=1)
#
# print('decoder_input_ids')
# print(decoder_input_ids)
# # Decode the generated token IDs into text
# summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#
# # Print the summary
# print(summary)
