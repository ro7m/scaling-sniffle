import 'dart:ui' as ui;
import 'dart:typed_data';
import '../constants.dart';

class ImagePreprocessor {
  // Preprocess image for detection
  Future<Float32List> preprocessForDetection(ui.Image image) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
      ui.Paint()
    );

    final picture = recorder.endRecording();
    final scaledImage = await picture.toImage(width, height);
    final byteData = await scaledImage.toByteData(format: ui.ImageByteFormat.rawRgba);

    if (byteData == null) {
      throw Exception('Failed to get byte data from image');
    }

    final pixels = byteData.buffer.asUint8List();
    final Float32List preprocessedData = Float32List(3 * width * height);

    for (int i = 0; i < pixels.length; i += 4) {
      final int idx = i ~/ 4;
      preprocessedData[idx] = (pixels[i] / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0]; // R
      preprocessedData[idx + width * height] = (pixels[i + 1] / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1]; // G
      preprocessedData[idx + 2 * width * height] = (pixels[i + 2] / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2]; // B
    }

    return preprocessedData;
  }

  // Preprocess image for recognition
Future<Map<String, dynamic>> preprocessImageForRecognition(
    List<ui.Image> crops, {
    List<int> targetSize = const [32, 128],
    List<double> mean = const [0.694, 0.695, 0.693],
    List<double> std = const [0.299, 0.296, 0.301],
  }) async {
    // Process each crop
    List<Float32List> processedImages = await Future.wait(
      crops.map((crop) => _processImage(crop, targetSize, mean, std))
    );

    // Concatenate multiple processed images if needed
    if (processedImages.length > 1) {
      final combinedLength = 3 * targetSize[0] * targetSize[1] * processedImages.length;
      final combinedData = Float32List(combinedLength);

      for (int i = 0; i < processedImages.length; i++) {
        combinedData.setAll(i * processedImages[i].length, processedImages[i]);
      }

      return {
        'data': combinedData,
        'dims': [processedImages.length, 3, targetSize[0], targetSize[1]],
      };
    }

    // Single image case
    return {
      'data': processedImages[0],
      'dims': [1, 3, targetSize[0], targetSize[1]],
    };
  }

  Future<Float32List> _processImage(
    ui.Image image,
    List<int> targetSize,
    List<double> mean,
    List<double> std,
  ) async {
    final targetHeight = targetSize[0];
    final targetWidth = targetSize[1];

    // Calculate resize dimensions
    double resizedWidth, resizedHeight;
    final aspectRatio = targetWidth / targetHeight;

    if (aspectRatio * image.height > image.width) {
      resizedHeight = targetHeight.toDouble();
      resizedWidth = ((targetHeight * image.width) / image.height).roundToDouble();
    } else {
      resizedWidth = targetWidth.toDouble();
      resizedHeight = ((targetWidth * image.height) / image.width).roundToDouble();
    }

    // Create a new canvas with black background
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    // Fill with black background
    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, targetWidth.toDouble(), targetHeight.toDouble()),
      ui.Paint()..color = ui.Color(0xFF000000),
    );

    // Calculate offsets for centering
    final xOffset = ((targetWidth - resizedWidth) / 2).floor().toDouble();
    final yOffset = ((targetHeight - resizedHeight) / 2).floor().toDouble();

    // Draw resized image
    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(xOffset, yOffset, resizedWidth, resizedHeight),
      ui.Paint(),
    );

    // Get the image data
    final picture = recorder.endRecording();
    final resizedImage = await picture.toImage(targetWidth, targetHeight);
    final byteData = await resizedImage.toByteData(format: ui.ImageByteFormat.rawRgba);

    if (byteData == null) {
      throw Exception('Failed to get byte data from image');
    }

    final pixels = byteData.buffer.asUint8List();
    final Float32List float32Data = Float32List(3 * targetHeight * targetWidth);

    // Normalize and separate channels
    for (int y = 0; y < targetHeight; y++) {
      for (int x = 0; x < targetWidth; x++) {
        final pixelIndex = (y * targetWidth + x) * 4;
        final channelSize = targetHeight * targetWidth;

        // Extract RGB and normalize
        final r = (pixels[pixelIndex] / 255.0 - mean[0]) / std[0];
        final g = (pixels[pixelIndex + 1] / 255.0 - mean[1]) / std[1];
        final b = (pixels[pixelIndex + 2] / 255.0 - mean[2]) / std[2];

        // Store normalized values in float32Data
        float32Data[y * targetWidth + x] = r;
        float32Data[channelSize + y * targetWidth + x] = g;
        float32Data[2 * channelSize + y * targetWidth + x] = b;
      }
    }

    return float32Data;
  }

}