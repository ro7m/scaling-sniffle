import 'dart:ui' as ui;
import 'dart:typed_data';
import '../models/ocr_result.dart';
import '../models/bounding_box.dart';
import 'model_loader.dart';
import 'image_preprocessor.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';
import 'dart:math' as math;

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

  double clamp(double number, double size) {
    return math.max(0, math.min(number, size));
  }

  BoundingBox transformBoundingBox(BoundingBox contour, int id, List<int> size) {
    double offset = (contour.width * contour.height * 1.8) / (2 * (contour.width + contour.height));
    double p1 = clamp(contour.x - offset, size[1].toDouble()) - 1;
    double p2 = clamp(p1 + contour.width + 2 * offset, size[1].toDouble()) - 1;
    double p3 = clamp(contour.y - offset, size[0].toDouble()) - 1;
    double p4 = clamp(p3 + contour.height + 2 * offset, size[0].toDouble()) - 1;

    return BoundingBox(
      x: p1 / size[1],
      y: p3 / size[0],
      width: (p2 - p1) / size[1],
      height: (p4 - p3) / size[0],
    );
  }

Future<List<OCRResult>> processImage(ui.Image image, {void Function(String)? debugCallback}) async {
  this.debugCallback = debugCallback;
  try {
    debugCallback?.call('Starting image processing...');

      final preprocessedImage = await imagePreprocessor.preprocessForDetection(image);
      debugCallback?.call('Image preprocessed for detection');

      final detectionResult = await textDetector!.runDetection(preprocessedImage);
      debugCallback?.call('Detection completed');

      // Use the new heatmap-based bounding box extraction
      final boundingBoxes = await textDetector!.processDetectionOutput(detectionResult);
      debugCallback?.call('Found ${boundingBoxes.length} bounding boxes');

    if (boundingBoxes.isEmpty) {
      return [];
    }

    // Transform bounding boxes to original image coordinates
    final transformedBoundingBoxes = boundingBoxes.map((box) => transformBoundingBox(box, 0, [image.height, image.width])).toList();

    final results = <OCRResult>[];
    for (var box in transformedBoundingBoxes) {
      final croppedImage = await _cropImage(image, box);
      final preprocessedCrop = await imagePreprocessor.preprocessForRecognition(croppedImage);
      final text = await textRecognizer!.recognizeText(preprocessedCrop, transformedBoundingBoxes.length); // Pass the number of crops
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
      box.x * image.width,
      box.y * image.height,
      box.width * image.width,
      box.height * image.height,
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