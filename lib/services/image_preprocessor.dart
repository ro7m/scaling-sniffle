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
  Future<Float32List> preprocessForRecognition(ui.Image image) async {
    final targetHeight = OCRConstants.REC_TARGET_SIZE[0];
    final targetWidth = OCRConstants.REC_TARGET_SIZE[1];

    double resizedWidth, resizedHeight;
    final aspectRatio = targetWidth / targetHeight;

    if (aspectRatio * image.height > image.width) {
      resizedHeight = targetHeight.toDouble();
      resizedWidth = (targetHeight * image.width) / image.height;
    } else {
      resizedWidth = targetWidth.toDouble();
      resizedHeight = (targetWidth * image.height) / image.width;
    }

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, targetWidth.toDouble(), targetHeight.toDouble()),
      ui.Paint()..color = ui.Color(0xFF000000),
    );

    final xOffset = (targetWidth - resizedWidth) / 2;
    final yOffset = (targetHeight - resizedHeight) / 2;
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
    final Float32List preprocessedData = Float32List(3 * targetWidth * targetHeight);

    for (int i = 0; i < pixels.length; i += 4) {
      final int idx = i ~/ 4;
      preprocessedData[idx] = (pixels[i] / 255.0 - OCRConstants.REC_MEAN[0]) / OCRConstants.REC_STD[0]; // R
      preprocessedData[idx + targetWidth * targetHeight] = (pixels[i + 1] / 255.0 - OCRConstants.REC_MEAN[1]) / OCRConstants.REC_STD[1]; // G
      preprocessedData[idx + 2 * targetWidth * targetHeight] = (pixels[i + 2] / 255.0 - OCRConstants.REC_MEAN[2]) / OCRConstants.REC_STD[2]; // B
    }

    return preprocessedData;
  }
}