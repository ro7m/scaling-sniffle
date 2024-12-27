import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class ModelLoader {
  OrtSession? detectionModel;
  OrtSession? recognitionModel;

  Future<void> loadModels({void Function(String)? debugCallback}) async {
    try {
      debugCallback?.call('Starting to load models...');
      final sessionOptions = OrtSessionOptions();

      debugCallback?.call('Loading detection model...');
      final detectionBytes = await rootBundle.load('assets/models/rep_fast_base.onnx');
      detectionModel = await OrtSession.fromBuffer(
        detectionBytes.buffer.asUint8List(
          detectionBytes.offsetInBytes,
          detectionBytes.lengthInBytes
        ),
        sessionOptions
      );
      debugCallback?.call('Detection model loaded: ${detectionBytes.lengthInBytes ~/ 1024}KB');

      debugCallback?.call('Loading recognition model...');
      final recognitionBytes = await rootBundle.load('assets/models/crnn_mobilenet_v3_large.onnx');
      recognitionModel = await OrtSession.fromBuffer(
        recognitionBytes.buffer.asUint8List(
          recognitionBytes.offsetInBytes,
          recognitionBytes.lengthInBytes
        ),
        sessionOptions
      );
      debugCallback?.call('Recognition model loaded: ${recognitionBytes.lengthInBytes ~/ 1024}KB');
    } catch (e) {
      debugCallback?.call('Error loading models: $e');
      throw Exception('Failed to load models: $e');
    }
  }
}