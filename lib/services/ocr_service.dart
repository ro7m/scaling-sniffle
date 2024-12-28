import 'dart:ui' as ui;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'model_loader.dart';
import 'image_preprocessor.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';
import '../models/bounding_box.dart';
import '../models/ocr_result.dart';

class OCRService {
  final ModelLoader modelLoader = ModelLoader();
  final ImagePreprocessor imagePreprocessor = ImagePreprocessor();
  TextDetector? textDetector;
  TextRecognizer? textRecognizer;
  void Function(String)? debugCallback;

  set setDebugCallback(void Function(String)? callback) {
    debugCallback = callback;
  }

  Future<void> loadModels({void Function(String)? debugCallback}) async {
    await cv.Cv2.initOpenCV();
    await modelLoader.loadModels(debugCallback: debugCallback);
    textDetector = TextDetector(modelLoader.detectionModel!);
    textRecognizer = TextRecognizer(modelLoader.recognitionModel!);
  }

  Future<List<OCRResult>> processImage(ui.Image image, {void Function(String)? debugCallback}) async {
    this.debugCallback = debugCallback;
    try {
      debugCallback?.call('Starting image processing...');

      final preprocessedImage = await imagePreprocessor.preprocessForDetection(image);
      debugCallback?.call('Image preprocessed for detection');

      final detectionResult = await textDetector!.runDetection(preprocessedImage);
      debugCallback?.call('Detection completed');

      final boundingBoxes = await textDetector!.processDetectionOutput(detectionResult);
      debugCallback?.call('Found ${boundingBoxes.length} bounding boxes');

      final results = <OCRResult>[];
      for (var box in boundingBoxes) {
        final croppedImage = await _cropImage(image, box);
        final preprocessedCrop = await imagePreprocessor.preprocessForRecognition(croppedImage);
        final text = await textRecognizer!.recognizeText(preprocessedCrop);
        if (text.isNotEmpty) {
          results.add(OCRResult(text: text, boundingBox: box));
        }
      }

      debugCallback?.call('Processed ${results.length} text regions');
      return results;
    } catch (e, stack) {
      debugCallback?.call('Error in processImage: $e\n$stack');
      throw Exception('Error in processImage: $e');
    }
  }

  Future<ui.Image> _cropImage(ui.Image image, BoundingBox box) async {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = Rect.fromLTWH(
      box.x * image.width,
      box.y * image.height,
      box.width * image.width,
      box.height * image.height,
    );
    
    canvas.drawImageRect(image, src, dst, Paint());
    return recorder.endRecording().toImage(
      (box.width * image.width).round(),
      (box.height * image.height).round(),
    );
  }
}