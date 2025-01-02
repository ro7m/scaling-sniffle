import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:math' as math;
import '../constants.dart';

class ImagePreprocessor {
Future<Float32List> preprocessForDetection(ui.Image image) async {
  final width = OCRConstants.TARGET_SIZE[0];
  final height = OCRConstants.TARGET_SIZE[1];

  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);

  // Just draw the image directly without setting any background
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

  // Match JavaScript pixel processing exactly
  for (int i = 0; i < pixels.length; i += 4) {
    final idx = i ~/ 4;
    preprocessedData[idx] = 
        (pixels[i] / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0];
    preprocessedData[idx + width * height] = 
        (pixels[i + 1] / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1];
    preprocessedData[idx + width * height * 2] = 
        (pixels[i + 2] / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2];
  }

  return preprocessedData;
}

  Future<Map<String, dynamic>> preprocessImageForRecognition(List<ui.Image> crops) async {
    const targetSize = [32, 128];  // [height, width]
    const mean = OCRConstants.REC_MEAN;  // Matching JS values
    const std = OCRConstants.REC_STD;   // Matching JS values

    List<Float32List> processedImages = await Future.wait(
      crops.map((crop) => _processImageForRecognition(
        crop, 
        targetSize,
        mean,
        std
      ))
    );

    // Match JavaScript batching logic
    if (processedImages.length > 1) {
      final combinedLength = 3 * targetSize[0] * targetSize[1] * processedImages.length;
      final Float32List combinedData = Float32List(combinedLength);

      for (int i = 0; i < processedImages.length; i++) {
        combinedData.setRange(i * processedImages[i].length, 
                            (i + 1) * processedImages[i].length, 
                            processedImages[i]);
      }

      return {
        'data': combinedData,
        'dims': [processedImages.length, 3, targetSize[0], targetSize[1]],
      };
    }

    return {
      'data': processedImages[0],
      'dims': [1, 3, targetSize[0], targetSize[1]],
    };
  }

  Future<Float32List> _processImageForRecognition(
    ui.Image image,
    List<int> targetSize,
    List<double> mean,
    List<double> std,
  ) async {
    final targetHeight = targetSize[0];
    final targetWidth = targetSize[1];
    
    // Calculate resize dimensions matching JavaScript logic
    double aspectRatio = targetWidth / targetHeight;
    double resizedWidth, resizedHeight;
    
    if (aspectRatio * image.height > image.width) {
      resizedHeight = targetHeight.toDouble();
      resizedWidth = ((targetHeight * image.width) / image.height).roundToDouble();
    } else {
      resizedWidth = targetWidth.toDouble();
      resizedHeight = ((targetWidth * image.height) / image.width).roundToDouble();
    }

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    // Fill with black background
    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, targetWidth.toDouble(), targetHeight.toDouble()),
      ui.Paint()..color = ui.Color(0xFF000000),
    );

    // Center the image
    double xOffset = ((targetWidth - resizedWidth) / 2).floorToDouble();
    double yOffset = ((targetHeight - resizedHeight) / 2).floorToDouble();

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(xOffset, yOffset, resizedWidth, resizedHeight),
      ui.Paint(),
    );

    final picture = recorder.endRecording();
    final resizedImage = await picture.toImage(targetWidth, targetHeight);
    final byteData = await resizedImage.toByteData(format: ui.ImageByteFormat.rawRgba);

    if (byteData == null) {
      throw Exception('Failed to get byte data from image');
    }

    final pixels = byteData.buffer.asUint8List();
    final Float32List preprocessedData = Float32List(3 * targetHeight * targetWidth);
    final channelSize = targetHeight * targetWidth;

    // Match JavaScript pixel processing exactly
    for (int y = 0; y < targetHeight; y++) {
      for (int x = 0; x < targetWidth; x++) {
        final int pixelIndex = (y * targetWidth + x) * 4;
        final int outputIndex = y * targetWidth + x;
        
        // RGB channel ordering
        preprocessedData[outputIndex] = 
            (pixels[pixelIndex] / 255.0 - mean[0]) / std[0];
        preprocessedData[channelSize + outputIndex] = 
            (pixels[pixelIndex + 1] / 255.0 - mean[1]) / std[1];
        preprocessedData[2 * channelSize + outputIndex] = 
            (pixels[pixelIndex + 2] / 255.0 - mean[2]) / std[2];
      }
    }

    return preprocessedData;
  }
}
