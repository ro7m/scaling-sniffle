import 'dart:typed_data';
import 'dart:math' as math;
import 'package:onnxruntime/onnxruntime.dart';
import '../constants.dart';

class TextRecognizer {
  final OrtSession recognitionModel;

  TextRecognizer(this.recognitionModel);

  Future<List<String>> recognizeTexts(Float32List preprocessedImages, int batchSize) async {
    final shape = [
      batchSize,  // Number of crops
      3, 
      OCRConstants.REC_TARGET_SIZE[0], 
      OCRConstants.REC_TARGET_SIZE[1]
    ];
    
    final inputOrt = OrtValueTensor.createTensorWithDataList(preprocessedImages, shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();

    List<OrtValue>? modelResults;
    try {
      final results = await recognitionModel.runAsync(runOptions, inputs);
      modelResults = results?.whereType<OrtValue>().toList();
      
      inputOrt.release();
      runOptions.release();

      if (modelResults == null || modelResults.isEmpty) {
        throw Exception('Recognition model output is null');
      }

      final output = modelResults.first.value as List;
      final List<double> logits = _flattenNestedList(output);

      final height = OCRConstants.REC_TARGET_SIZE[0];
      final numClasses = OCRConstants.VOCAB.length + 1; // +1 for blank token
      final List<String> decodedTexts = [];

      // Process each image in the batch
      for (int b = 0; b < batchSize; b++) {
        final List<List<double>> probabilities = [];
        
        // Process logits for each timestep for this image
        for (int h = 0; h < height; h++) {
          final int baseIndex = b * height * numClasses + h * numClasses;
          final List<double> timestepLogits = logits.sublist(
            baseIndex, 
            baseIndex + numClasses
          );
          probabilities.add(_softmax(timestepLogits));
        }

        // Find best path using greedy decoding
        final List<int> bestPath = [];
        int prevClass = -1;
        
        for (int h = 0; h < height; h++) {
          final List<double> probs = probabilities[h];
          final int bestClass = _argmax(probs);
          
          // Apply CTC decoding rules
          if (bestClass != numClasses - 1 && bestClass != prevClass) {
            bestPath.add(bestClass);
            prevClass = bestClass;
          }
        }

        // Convert indices to text
        final StringBuffer decodedText = StringBuffer();
        for (final index in bestPath) {
          if (index < OCRConstants.VOCAB.length) {
            decodedText.write(OCRConstants.VOCAB[index]);
          }
        }
        
        decodedTexts.add(decodedText.toString());
      }

      // Release memory
      modelResults.forEach((element) {
        if (element != null) element.release();
      });

      return decodedTexts;
    } catch (e) {
      modelResults?.forEach((element) {
        if (element != null) element.release();
      });
      throw Exception('Error in text recognition: $e');
    }
  }

  // Helper function to compute softmax
  List<double> _softmax(List<double> logits) {
    // Find max for numerical stability
    final double maxLogit = logits.reduce(math.max);
    
    // Compute exp of shifted logits
    final List<double> expLogits = logits.map(
      (x) => math.exp(x - maxLogit)
    ).toList();
    
    // Compute sum for normalization
    final double sumExp = expLogits.reduce((a, b) => a + b);
    
    // Normalize to get probabilities
    return expLogits.map((x) => x / sumExp).toList();
  }

  // Helper function to find argmax
  int _argmax(List<double> array) {
    int maxIndex = 0;
    double maxValue = array[0];
    
    for (int i = 1; i < array.length; i++) {
      if (array[i] > maxValue) {
        maxValue = array[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
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
