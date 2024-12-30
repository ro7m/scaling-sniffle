import 'dart:typed_data';
import 'dart:math' as math;
import 'package:onnxruntime/onnxruntime.dart';
import '../constants.dart';

class TextRecognizer {
  final OrtSession recognitionModel;

  TextRecognizer(this.recognitionModel);

  Future<List<String>> recognizeText(Map<String, dynamic> preprocessedData) async {
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      preprocessedData['data'] as Float32List,
      preprocessedData['dims'] as List<int>
    );
    
    final inputs = {'input': inputTensor};
    final runOptions = OrtRunOptions();

    try {
      final results = await recognitionModel.runAsync(runOptions, inputs);
      if (results == null || results.isEmpty) {
        throw Exception('Recognition model output is null');
      }

      final output = results.first?.value;
      if (output == null) {
        throw Exception('Recognition model output is null');
      }

      // Get dimensions from model output
      final dimensions = results.first!.tensorData.dimensions;
      final [batchSize, height, numClasses] = dimensions;

      // Convert output to logits
      final logits = _flattenNestedList(output as List);
      
      // Process each batch item
      final List<String> decodedTexts = [];
      for (int b = 0; b < batchSize; b++) {
        final batchLogits = _extractBatchLogits(logits, b, height, numClasses);
        final text = _decodeLogits(batchLogits, height, numClasses);
        decodedTexts.add(text);
      }

      return decodedTexts;
    } finally {
      inputTensor.release();
      runOptions.release();
      results?.forEach((element) => element?.release());
    }
  }

  List<double> _extractBatchLogits(List<double> logits, int batchIndex, int height, int numClasses) {
    final batchLogits = <double>[];
    final batchOffset = batchIndex * height * numClasses;
    
    for (int h = 0; h < height; h++) {
      final startIdx = batchOffset + (h * numClasses);
      final endIdx = startIdx + numClasses;
      batchLogits.addAll(logits.sublist(startIdx, endIdx));
    }
    
    return batchLogits;
  }

  String _decodeLogits(List<double> logits, int height, int numClasses) {
    final StringBuffer decodedText = StringBuffer();
    int prevIndex = -1;

    for (int h = 0; h < height; h++) {
      final List<double> timestepLogits = logits.sublist(h * numClasses, (h + 1) * numClasses);
      final softmaxed = _softmax(timestepLogits);
      final maxIndex = softmaxed.indexOf(softmaxed.reduce(math.max));
      
      // CTC decoding logic - skip blank token (last class) and repeated characters
      if (maxIndex != numClasses - 1 && maxIndex != prevIndex) {
        if (maxIndex < OCRConstants.VOCAB.length) {
          decodedText.write(OCRConstants.VOCAB[maxIndex]);
        }
      }
      prevIndex = maxIndex;
    }

    return decodedText.toString();
  }

  List<double> _softmax(List<double> logits) {
    // Numerical stability by subtracting max
    final maxLogit = logits.reduce(math.max);
    final expLogits = logits.map((x) => math.exp(x - maxLogit)).toList();
    final sumExp = expLogits.reduce((a, b) => a + b);
    return expLogits.map((x) => x / sumExp).toList();
  }

  Float32List _flattenNestedList(List nestedList) {
    final List<double> flattened = [];
    void flatten(dynamic item) {
      if (item is List) {
        for (var subItem in item) {
          flatten(subItem);
        }
      } else if (item is num) {
        flattened.add(item.toDouble());
      }
    }
    flatten(nestedList);
    return Float32List.fromList(flattened);
  }
}