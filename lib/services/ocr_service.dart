import 'dart:ui' as ui;
import '../models/ocr_result.dart';
import '../models/bounding_box.dart';
import 'model_loader.dart';
import 'image_preprocessor.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';

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

      final boundingBoxes = await textDetector!.extractBoundingBoxes(detectionResult, debugCallback: debugCallback);
      debugCallback?.call('Found ${boundingBoxes.length} bounding boxes');

      if (boundingBoxes.isEmpty) {
        return [];
      }

      // Transform bounding boxes to original image coordinates
      final transformedBoundingBoxes = transformBoundingBoxes(boundingBoxes, image.width, image.height, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]);

      final results = <OCRResult>[];
      for (var box in transformedBoundingBoxes) {
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

    final srcRect = ui.Rect.fromLTWH(
      box.x,
      box.y,
      box.width,
      box.height,
    );

    const targetHeight = 32.0;
    const targetWidth = 128.0;

    double resizedWidth, resizedHeight;
    final aspectRatio = targetWidth / targetHeight;

    if (aspectRatio * srcRect.height > srcRect.width) {
      resizedHeight = targetHeight;
      resizedWidth = (targetHeight * srcRect.width) / srcRect.height;
    } else {
      resizedWidth = targetWidth;
      resizedHeight = (targetWidth * srcRect.height) / srcRect.width;
    }

    final xPad = (targetWidth - resizedWidth) / 2;
    final yPad = (targetHeight - resizedHeight) / 2;

    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, targetWidth, targetHeight),
      ui.Paint()..color = ui.Color(0xFF000000),
    );

    canvas.drawImageRect(
      image,
      srcRect,
      ui.Rect.fromLTWH(xPad, yPad, resizedWidth, resizedHeight),
      ui.Paint(),
    );

    final picture = recorder.endRecording();
    return await picture.toImage(targetWidth.toInt(), targetHeight.toInt());
  }
}