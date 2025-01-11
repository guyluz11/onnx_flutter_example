import json

# Path to the JSON file containing special tokens
json_file_path = "long_t5/special_tokens_map.json"
# Output Dart file path
dart_file_path = "long_t5/special_tokens.dart"

# Load the JSON file and parse its content
def load_special_tokens(json_file_path):
    with open(json_file_path, "r") as json_file:
        special_tokens_map = json.load(json_file)

    # Prepare Dart map content
    dart_map = "Map<String, int> specialTokens = {\n"

    # Counter for assigning IDs
    id_counter = 100000  # Starting ID

    # Add additional special tokens
    additional_tokens = special_tokens_map.get("additional_special_tokens", [])
    for token in additional_tokens:
        dart_map += f'  "{token}": {id_counter},\n'
        id_counter += 1

    # Add specific special tokens
    for key in ["eos_token", "pad_token", "unk_token"]:
        if key in special_tokens_map:
            dart_map += f'  "{special_tokens_map[key]}": {id_counter},\n'
            id_counter += 1

    dart_map += "};\n"

    return dart_map

# Write the Dart map to a Dart file
def write_dart_file(dart_map, dart_file_path):
    with open(dart_file_path, "w") as dart_file:
        dart_file.write(dart_map)

# Main script logic
if __name__ == "__main__":
    dart_map_content = load_special_tokens(json_file_path)
    write_dart_file(dart_map_content, dart_file_path)
    print(f"Dart file with special tokens saved to: {dart_file_path}")
