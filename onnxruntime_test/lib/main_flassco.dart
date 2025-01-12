import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:langchain_tiktoken/langchain_tiktoken.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime_test/helpers/texts.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  OrtEnv.instance.init();
  runApp(MaterialApp(
    home: TextSummarization(),
  ));
}

class TextSummarization extends StatefulWidget {
  @override
  _TextSummarizationState createState() => _TextSummarizationState();
}

class _TextSummarizationState extends State<TextSummarization> {
  String _summary = "";

  Future<OrtSession> loadDecoderSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/flassco_decoder_model.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    return OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<OrtSession> loadEncoderSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/flassco_encoder_model.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    return OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<String?> summery(String text) async {
    Tiktoken tiktoken = encodingForModel("t5");
    Uint32List encodedUintList = tiktoken.encode(text);

    // print('encodedList');
    // print(encodedList);
    List<List<int>> inputList = [encodedUintList.toList()];

    List<List<int>> attentionMask = createAttentionMask(inputList);

    OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(inputList);
    OrtValueTensor attentionMaskOrt =
        OrtValueTensor.createTensorWithDataList(attentionMask);
    OrtRunOptions runOptions = OrtRunOptions();
    print('Start');
    List<List<List<double>>>? outputs = await generatEncode(
      attentionMaskOrt: attentionMaskOrt,
      inputOrt: inputOrt,
      runOptions: runOptions,
    );

    if (outputs == null) {
      print('There was error decoding');
      return null;
    }
    List<List<Float32List>> floatOutputs = _convertToFloat32(outputs);

    OrtValueTensor encodeOutput =
        OrtValueTensor.createTensorWithDataList(floatOutputs);

    int maxSummaryLength = 70; // Maximum tokens for the summary
    int eosTokenId = 1; // Example [EOS] token ID (replace with your actual ID)

    // print(outputs);
    List<int>? decodeInts = await generateDecode(
      attentionMaskOrt: attentionMaskOrt,
      inputOrt: inputOrt,
      runOptions: runOptions,
      encodeOutput: encodeOutput,
      maxSummaryLength: maxSummaryLength,
      eosTokenId: eosTokenId,
    );

    if (decodeInts == null) {
      print('There was error decodeInts');
      return null;
    }

    String decodeString = tiktoken.decode(decodeInts);

    inputOrt.release();
    attentionMaskOrt.release();
    runOptions.release();
    encodeOutput.release();

    return decodeString;
  }

  /// Helper function to flatten and convert nested List<List<List<double>>> to Float32List
  List<List<Float32List>> _convertToFloat32(List<List<List<double>>> data) {
    // Convert the nested list to a nested Float32List
    return data
        .map((outerList) => outerList
            .map((innerList) => Float32List.fromList(innerList))
            .toList())
        .toList();
  }

  List<List<int>> createAttentionMask(List<List<int>> inputList) {
    // Iterate over each inner list to create the attention mask
    return inputList.map((sequence) {
      return sequence.map((token) => token != 0 ? 1 : 0).toList();
    }).toList();
  }

  Future<List<List<List<double>>>?> generatEncode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtRunOptions runOptions,
  }) async {
    List<OrtValue?>? outputs;

    final inputs = {
      'input_ids': inputOrt,
      'attention_mask': attentionMaskOrt,
    };

    print('Start generatEncode');
    OrtSession session = await loadEncoderSession();
    outputs = await session.runAsync(runOptions, inputs);
    print('Done generatEncode');

    if (outputs == null || outputs.isEmpty) {
      return null;
    }

    OrtValue? output0 = outputs[0];
    if (output0 == null) {
      return null;
    }
    List<List<List<double>>> output0Value =
        output0.value as List<List<List<double>>>;

    outputs.forEach((element) {
      element?.release();
    });
    session.release();

    return output0Value;
  }

  Future<List<int>?> generateDecode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtValueTensor encodeOutput,
    required OrtRunOptions runOptions,
    required int maxSummaryLength, // Maximum summary length
    required int eosTokenId, // End-of-sequence token ID
  }) async {
    OrtSession session = await loadDecoderSession();
    List<int> currentOutput = []; // Stores the generated token IDs

    // Start with the initial decoder input (e.g., [BOS] token ID)
    List<int> initialDecoderInput = [
      inputOrt.value.first[0]
    ]; // Assuming the first token
    currentOutput.addAll(initialDecoderInput);

    print('Start generateDecode');

    // Iterate up to the maximum summary length
    for (int i = 0; i < maxSummaryLength; i++) {
      // Prepare inputs for the decoder
      final inputs = {
        'input_ids': OrtValueTensor.createTensorWithDataList([currentOutput]),
        'encoder_attention_mask': attentionMaskOrt,
        'encoder_hidden_states': encodeOutput,
      };

      // Run the decoder
      List<OrtValue?>? outputs = await session.runAsync(runOptions, inputs);

      if (outputs == null || outputs.isEmpty) {
        print('Decoder outputs are empty!');
        break;
      }

      // Extract logits and find the next token
      OrtValue? output0 = outputs[0];
      if (output0 == null) {
        print('Decoder output[0] is null!');
        break;
      }
      List<List<List<double>>> output0Value =
          output0.value as List<List<List<double>>>;
      List<int> nextTokenIds = npArgmax(output0Value[0]);
      int nextTokenId = nextTokenIds.last; // Get the last token ID

      // Append the new token to the current output
      currentOutput.add(nextTokenId);

      // Release outputs to free resources
      outputs.forEach((element) {
        element?.release();
      });

      // Stop if the [EOS] token is generated
      if (nextTokenId == eosTokenId) {
        print('EOS token encountered. Stopping decoding.');
        break;
      }
    }
    print('Done generateDecode');

    session.release();

    return currentOutput;
  }

  List<List<int>> npConcatenate(
      List<List<int>> decoderInputIds, List<List<int>> newToken) {
    // Ensure both lists are non-empty and have compatible dimensions
    if (decoderInputIds.isEmpty) {
      return newToken;
    }

    for (int i = 0; i < decoderInputIds.length; i++) {
      decoderInputIds[i].addAll(newToken[i]);
    }

    return decoderInputIds;
  }

  Future<void> flasscoSummarize(String inputText) async {
    String preprocessTextVar = preprocessText(inputText);
    String? summeryOutput = await summery(inputText);

    setState(() {
      _summary = summeryOutput ?? 'Error summarizing';
    });
  }

  /// Using axis -1
  List<int> npArgmax(List<List<double>> logits) {
    List<int> maxIndices = [];

    for (List<double> innerList in logits) {
      int maxIndex = 0;
      double maxValue = innerList[0];

      for (int i = 1; i < innerList.length; i++) {
        if (innerList[i] > maxValue) {
          maxValue = innerList[i];
          maxIndex = i;
        }
      }

      maxIndices.add(maxIndex);
    }

    return maxIndices;
  }

  String preprocessText(String text) {
    // Remove consecutive punctuation (e.g., `...,`)
    text = text.replaceAll(RegExp(r'[,.]{2,}'), ' ');

    // Remove extra spaces
    text = text.replaceAll(RegExp(r'\s+'), ' ').trim();

    // Remove non-ASCII characters
    text = text.replaceAll(RegExp(r'[^\x00-\x7F]+'), ' ');

    // Lowercase the text (optional)
    text = text.toLowerCase();

    return text;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Text Summarization with ONNX'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              onChanged: (text) {
                setState(() {
                  _summary = "Generating summary...";
                });
                flasscoSummarize('your');
              },
              decoration: InputDecoration(
                hintText: "Enter text to summarize",
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 20),
            TextButton(
              onPressed: () => flasscoSummarize(getLongText),
              child: Text('Press to impress'),
            ),
            SizedBox(height: 20),
            Text('Summary: $_summary'),
          ],
        ),
      ),
    );
  }
}
