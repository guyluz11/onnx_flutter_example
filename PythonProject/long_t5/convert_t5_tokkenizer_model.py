
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2
import base64

# Load the SentencePiece model file
spiece_model_path = "spiece.model"

# Function to parse SentencePiece model and convert to Dart map
def convert_sentencepiece_to_dart(spiece_model_path):
    # Load the model as a protobuf object
    model_proto = sentencepiece_model_pb2.ModelProto()
    with open(spiece_model_path, "rb") as file:
        model_proto.ParseFromString(file.read())

    # Extract tokens and their IDs with base64-encoded keys
    vocab_bytecode = {
        base64.b64encode(piece.piece.encode("utf-8")).decode("utf-8"): idx
        for idx, piece in enumerate(model_proto.pieces)
    }
    vocab_string = {
        piece.piece: idx for idx, piece in enumerate(model_proto.pieces)
    }

    # Prepare Dart-compatible map format for bytecode
    dart_map_format_bytecode = "final Map<String, int> t5BaseMapBytecode = {\n"
    dart_map_format_bytecode += ",\n".join(f'  "{token}": {id}' for token, id in vocab_bytecode.items())
    dart_map_format_bytecode += "\n};"

    # Prepare Dart-compatible map format for strings
    dart_map_format_string = "final Map<String, int> t5BaseMapString = {\n"
    dart_map_format_string += ",\n".join(f'  "{token}": {id}' for token, id in vocab_string.items())
    dart_map_format_string += "\n};"

    # Save the Dart maps to separate files
    output_path_bytecode = "spiece_model_dart_map_bytecode.dart"
    output_path_string = "spiece_model_dart_map_string.dart"

    with open(output_path_bytecode, "w") as output_file:
        output_file.write(dart_map_format_bytecode)

    with open(output_path_string, "w") as output_file:
        output_file.write(dart_map_format_string)

    return output_path_bytecode, output_path_string

# Convert the model and get the output paths
output_path_bytecode, output_path_string = convert_sentencepiece_to_dart(spiece_model_path)
print(f"Dart maps saved to: {output_path_bytecode} and {output_path_string}")
