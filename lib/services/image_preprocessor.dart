import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:math'; 
import '../constants.dart';

class ImagePreprocessor {
  Future<Float32List> preprocessForDetection(ui.Image image) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    // Maintain aspect ratio while resizing
    double scale = math.min(
      width / image.width,
      height / image.height
    );
    double newWidth = image.width * scale;
    double newHeight = image.height * scale;
    
    // Center the image
    double offsetX = (width - newWidth) / 2;
    double offsetY = (height - newHeight) / 2;

    // Fill with white background first
    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
      ui.Paint()..color = ui.Color(0xFFFFFFFF)
    );

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(offsetX, offsetY, newWidth, newHeight),
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

    // NCHW format: [batch, channel, height, width]
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixelIndex = (y * width + x) * 4;
        final int outputIndex = y * width + x;
        
        // Channel ordering: RGB
        preprocessedData[outputIndex] = 
            (pixels[pixelIndex] / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0];
        preprocessedData[width * height + outputIndex] = 
            (pixels[pixelIndex + 1] / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1];
        preprocessedData[2 * width * height + outputIndex] = 
            (pixels[pixelIndex + 2] / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2];
      }
    }

    return preprocessedData;
  }

  Future<Map<String, dynamic>> preprocessForRecognition(List<ui.Image> crops) async {
    const targetHeight = 32;
    const targetWidth = 128;
    const mean = [0.485, 0.456, 0.406]; // ImageNet mean
    const std = [0.229, 0.224, 0.225];  // ImageNet std

    List<Float32List> processedImages = await Future.wait(
      crops.map((crop) => _processImageForRecognition(
        crop, 
        targetHeight, 
        targetWidth,
        mean,
        std
      ))
    );

    // Batch the processed images
    final int batchSize = processedImages.length;
    final int channelSize = targetHeight * targetWidth;
    final int imageSize = 3 * channelSize;
    final Float32List batchedData = Float32List(batchSize * imageSize);

    for (int i = 0; i < batchSize; i++) {
      batchedData.setAll(i * imageSize, processedImages[i]);
    }

    return {
      'data': batchedData,
      'dims': [batchSize, 3, targetHeight, targetWidth], // NCHW format
    };
  }

  Future<Float32List> _processImageForRecognition(
    ui.Image image,
    int targetHeight,
    int targetWidth,
    List<double> mean,
    List<double> std,
  ) async {
    // Calculate resize dimensions maintaining aspect ratio
    double scale = math.min(
      targetWidth / image.width,
      targetHeight / image.height
    );
    double newWidth = (image.width * scale).roundToDouble();
    double newHeight = (image.height * scale).roundToDouble();

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    // Fill with white background
    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, targetWidth.toDouble(), targetHeight.toDouble()),
      ui.Paint()..color = ui.Color(0xFFFFFFFF),
    );

    // Center the image
    double offsetX = (targetWidth - newWidth) / 2;
    double offsetY = (targetHeight - newHeight) / 2;

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(offsetX, offsetY, newWidth, newHeight),
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

    // Convert to NCHW format with proper normalization
    for (int y = 0; y < targetHeight; y++) {
      for (int x = 0; x < targetWidth; x++) {
        final int pixelIndex = (y * targetWidth + x) * 4;
        final int outputIndex = y * targetWidth + x;
        
        // RGB channel ordering
        preprocessedData[outputIndex] = 
            (pixels[pixelIndex] / 255.0 - mean[0]) / std[0];
        preprocessedData[targetHeight * targetWidth + outputIndex] = 
            (pixels[pixelIndex + 1] / 255.0 - mean[1]) / std[1];
        preprocessedData[2 * targetHeight * targetWidth + outputIndex] = 
            (pixels[pixelIndex + 2] / 255.0 - mean[2]) / std[2];
      }
    }

    return preprocessedData;
  }
}
