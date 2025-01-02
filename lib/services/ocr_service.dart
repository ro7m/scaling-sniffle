import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:image_picker/image_picker.dart';
import 'model_loader.dart';
import 'image_preprocessor.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';
import '../models/ocr_result.dart';
import '../models/bounding_box.dart';
import 'dart:io';

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

  Future<List<OCRResult>> processImage(XFile imageFile, {void Function(String)? debugCallback}) async {
    this.debugCallback = debugCallback;
    try {
      debugCallback?.call('Starting image processing...');

      // Load image from file
      final File file = File(imageFile.path);
      final Uint8List bytes = await file.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(bytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();
      final ui.Image image = frameInfo.image;

      final preprocessedImage = await imagePreprocessor.preprocessForDetection(image);
      debugCallback?.call('Image preprocessed for detection');

      final detectionResult = await textDetector!.runDetection(preprocessedImage);
      debugCallback?.call('Detection completed');

      final boundingBoxes = await textDetector!.processImage(detectionResult);
      debugCallback?.call('Found ${boundingBoxes.length} bounding boxes');

      if (boundingBoxes.isEmpty) {
        return [];
      }

  final results = <OCRResult>[];
  final List<ui.Image> crops = [];
  
  for (var box in boundingBoxes) {
    try {
      final croppedImage = await _cropImage(image, box);
      crops.add(croppedImage);
    } catch (e) {
      debugCallback?.call('Error cropping image: $e');
    }
  }

  if (crops.isNotEmpty) {
    final preprocessed = await imagePreprocessor.preprocessImageForRecognition(crops);
    final texts = await textRecognizer!.recognizeText(preprocessed['data'] as Float32List,crops.length);
    
    for (int i = 0; i < texts.length; i++) {
      if (texts[i].isNotEmpty) {
        results.add(OCRResult(text: texts[i], boundingBox: boundingBoxes[i]));
      }
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
    return await picture.toImage(targetWidth.round(), targetHeight.round());
  }
}