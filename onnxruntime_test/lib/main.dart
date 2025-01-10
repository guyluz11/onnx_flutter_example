import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

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
  late OrtSession flasscoEncoderSession;
  late OrtSession flasscoDecoderSession;
  late OrtSession longT5Session;
  String _summary = "";

  @override
  void initState() {
    super.initState();
    loadLongT5Session();
    // loadEncoderSession();
    // loadDecoderSession();
  }

  Future<void> loadLongT5Session() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/long_t5.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    longT5Session = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<List<OrtValue?>?> summery(String text, OrtSession session) async {
    // Uint32List a = encoding.encode(text);
    // print(a);

    // List<int> tokenizedInput = _tokenizeText(inputText);
    List<List<int>> inputList = [
      [
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
        1,
      ]
    ];

    List<List<int>> attentionMask = [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ];
    List<List<int>> decoderInputIds = [
      [0]
    ];

    OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(inputList);
    OrtValueTensor attentionMaskOrt =
        OrtValueTensor.createTensorWithDataList(attentionMask);
    OrtValueTensor decoderInputIdsOrt =
        OrtValueTensor.createTensorWithDataList(decoderInputIds);
    OrtRunOptions runOptions = OrtRunOptions();
    List<int>? outputs = await generateSummery(
      attentionMaskOrt: attentionMaskOrt,
      decoderInputIdsOrt: decoderInputIdsOrt,
      inputOrt: inputOrt,
      session: session,
      runOptions: runOptions,
    );

    print(outputs);

    inputOrt.release();
    attentionMaskOrt.release();
    decoderInputIdsOrt.release();
    runOptions.release();

    return null;
    // Post-process the output
    // String summary = _decodeSummary(outputs);

    // setState(() {
    //   _summary = summary;
    // });
  }

  Future<List<int>?> generateSummery({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtValueTensor decoderInputIdsOrt,
    required OrtSession session,
    required OrtRunOptions runOptions,
  }) async {
    int outputMaxLength = 512;
    List<OrtValue?>? outputs;
    List<int> generatedTokens = [0];

    for (int a = 0; a < outputMaxLength; a++) {
      final inputs = {
        'input_ids': inputOrt,
        'attention_mask': attentionMaskOrt,
        'decoder_input_ids': decoderInputIdsOrt,
      };

      outputs = await session.runAsync(runOptions, inputs);

      if (outputs == null || outputs.isEmpty) {
        return null;
      }

      OrtValue? output0 = outputs[0];
      if (output0 == null) {
        return null;
      }
      List<List<List<double>>> output0Value =
          output0.value as List<List<List<double>>>;
      int summaryIds = npArgmax(output0Value[0][output0Value[0].length - 1]);
      generatedTokens.add(summaryIds);
      const tokenizerEosTokenId = 1;
      if (summaryIds == tokenizerEosTokenId) {
        break;
      }
      decoderInputIdsOrt =
          OrtValueTensor.createTensorWithDataList([generatedTokens]);
      // String summeryText = '';
      // // summeryText = encoding.decode(summaryIds[0]);
      //
      // print('Summery');
      // print(summeryText);
      // print('');

      outputs.forEach((element) {
        element?.release();
      });
    }
    return generatedTokens;
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
    List<OrtValue?>? encoderOutput =
        await summery(inputText, flasscoEncoderSession);
    if (encoderOutput?[0] == null) {
      print('No object');
      return;
    }
    List<OrtValue> nonNullEncoderOutput =
        (encoderOutput ?? []).whereType<OrtValue>().toList();
    flasscoDecoder(nonNullEncoderOutput);
    print('Done');
  }

  Future<String?> flasscoDecoder(List<OrtValue> encoded) async {
    print('input names');
    print(flasscoDecoderSession.inputNames);
    print('');

    // Pass the encoder hidden states directly (encoded)
    List<List<double>> encoderHiddenStatesOrt =
        (encoded[0].value as List<List<double>>);

    OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(encoderHiddenStatesOrt);

    // Create the inputs map for the decoder
    final inputs = {
      'encoder_hidden_states': inputOrt,
    };

    final runOptions = OrtRunOptions();

    List<OrtValue?>? outputs =
        await flasscoEncoderSession.runAsync(runOptions, inputs);
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) {
      element?.release();
    });
  }

  Future<void> loadDecoderSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/flassco_decoder_model.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    flasscoDecoderSession = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<void> loadEncoderSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/flassco_encoder_model.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    flasscoEncoderSession = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  // List<int> _tokenizeText(String text) {
  //   // Tokenize the input text (this example assumes a simple space-based tokenization)
  //   // You should ideally use a tokenizer like Hugging Face's tokenizer for proper tokenization
  //   return text.split(' ').map((word) => word.hashCode).toList();
  // }

// // Function to decode the token IDs to text
//   String decodeSummary(List<int> tokenIds, {bool skipSpecialTokens = true}) {
//     List<String> decodedTokens = [];
//     for (int tokenId in tokenIds) {
//       if (skipSpecialTokens && (tokenId == 0 || tokenId == 3 || tokenId == 4)) {
//         // Skip special tokens (like <BOS>, <EOS>, <PAD>)
//         continue;
//       }
//       decodedTokens.add(
//           tokenizerDict[tokenId] ?? '[UNK]'); // Use [UNK] for unknown token IDs
//     }
//     return decodedTokens.join(' ');
//   }

  /// Using axis -1
  int npArgmax(List<double> logits) {
    int maxIndex = 0;
    double maxValue = logits[0];

    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > maxValue) {
        maxValue = logits[i];
        maxIndex = i;
      }
    }

    return maxIndex;
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
              onPressed: () => summery(
                  "Here is a lot of text I don't want to read ok?",
                  longT5Session),
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
