import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:langchain_tiktoken/langchain_tiktoken.dart';
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
    Tiktoken tiktoken = encodingForModel("t5");
    Uint32List encodedList = tiktoken.encode(text);

    print('encodedList');
    print(encodedList);
    print(tiktoken.decode(encodedList));

    List<List<int>> inputList = [encodedList.toList()];
    // [
    //   [
    //     947,
    //     19,
    //     3,
    //     9,
    //     418,
    //     13,
    //     1499,
    //     27,
    //     278,
    //     31,
    //     17,
    //     241,
    //     12,
    //     608,
    //     3,
    //     1825,
    //     58,
    //     1,
    //   ]
    // ];

    List<List<int>> attentionMask = createAttentionMask(inputList);
    // [
    //   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    // ];
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
    print('Start');
    List<int>? outputs = await generateSummery(
      attentionMaskOrt: attentionMaskOrt,
      decoderInputIdsOrt: decoderInputIdsOrt,
      inputOrt: inputOrt,
      session: session,
      runOptions: runOptions,
    );
    if (outputs == null) {
      print('There was error decoding');
      return null;
    }

    print(outputs);
    String decodeString = tiktoken.decode(outputs);

    print('decodeString');
    print(decodeString);

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

  List<List<int>> createAttentionMask(List<List<int>> inputList) {
    // Iterate over each inner list to create the attention mask
    return inputList.map((sequence) {
      return sequence.map((token) => token != 0 ? 1 : 0).toList();
    }).toList();
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
              onPressed: () {
                String longText =
                    "Here is a lot of text I don't want to read ok?";
                //                    '''New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
                // A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
                // Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
                // In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
                // Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
                // 2010 marriage license application, according to court documents.
                // Prosecutors said the marriages were part of an immigration scam.
                // On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
                // After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
                // Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
                // All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
                // Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
                // Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
                // The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
                // Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
                // Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
                // If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
                // ''';
                summery(longText, longT5Session);
              },
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
