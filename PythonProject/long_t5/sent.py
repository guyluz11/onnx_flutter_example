import os

import sentencepiece as spm

# Load the SentencePiece model
model_path = 'local_model_dir/spiece.model'  # Update with the actual path to your model



# local_model_dir = model_path
# required_files = ["spiece.model", "tokenizer_config.json", "special_tokens_map.json"]
#
# for file in required_files:
#     if not os.path.exists(os.path.join(local_model_dir, file)):
#         print(f"Missing file: {file}")

sp = spm.SentencePieceProcessor(model_file=model_path)

# Search for token ID 19
token_id = 19

if token_id < 0 or token_id >= sp.vocab_size():
    print(f"Error: Token ID {token_id} is out of range or invalid.")
else:
    token = sp.id_to_piece(token_id)
    print(f"Token ID {token_id}: {token}")
