import 'dart:typed_data';
import 'dart:math' as math;
import 'package:onnxruntime/onnxruntime.dart';
import '../constants.dart';

class TextRecognizer {
  final OrtSession recognitionModel;

  TextRecognizer(this.recognitionModel);

  Future<String> recognizeText(Float32List preprocessedImage) async {
    final shape = [1, 3, OCRConstants.REC_TARGET_SIZE[0], OCRConstants.REC_TARGET_SIZE[1]];
    final inputOrt = OrtValueTensor.createTensorWithDataList(preprocessedImage, shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();

    final results = await recognitionModel.runAsync(runOptions, inputs);
    inputOrt.release();
    runOptions.release();

    final output = results.first.value;
    if (output == null) {
      throw Exception('Recognition model output is null');
    }
    final logits = _flattenNestedList(output as List);

    final batchSize = 1;
    final height = OCRConstants.REC_TARGET_SIZE[0];
    final numClasses = OCRConstants.VOCAB.length;

    List<double> softmax(List<double> logits) {
      final expLogits = logits.map((x) => math.exp(x)).toList();
      final sumExpLogits = expLogits.reduce((a, b) => a + b);
      return expLogits.map((x) => x / sumExpLogits).toList();
    }

    final List<int> bestPath = [];
    for (int h = 0; h < height; h++) {
      final List<double> timestepLogits = logits.sublist(h * numClasses, (h + 1) * numClasses);
      final softmaxed = softmax(timestepLogits);
      final maxIndex = softmaxed.indexWhere((x) => x == softmaxed.reduce(math.max));
      bestPath.add(maxIndex);
    }

    final StringBuffer decodedText = StringBuffer();
    int prevIndex = -1;
    for (final index in bestPath) {
      if (index != numClasses - 1 && index != prevIndex) {
        decodedText.write(OCRConstants.VOCAB[index]);
      }
      prevIndex = index;
    }

    results.forEach((element) {
      element.release();
    });

    return decodedText.toString();
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