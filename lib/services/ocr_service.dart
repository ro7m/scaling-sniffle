import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img_lib;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import '../constants.dart';
import '../models/bounding_box.dart';
import '../models/ocr_result.dart';

class _BoundingRegion {
  int minX = 999999;
  int maxX = -999999;
  int minY = 999999;
  int maxY = -999999;

  void update(int x, int y) {
    minX = math.min(minX, x);
    maxX = math.max(maxX, x);
    minY = math.min(minY, y);
    maxY = math.max(maxY, y);
  }
}

class OCRService {
  OrtSession? detectionModel;
  OrtSession? recognitionModel;
  
  Future<void> loadModels() async {
    try {
      final sessionOptions = OrtSessionOptions();
      final appDir = await getApplicationDocumentsDirectory();
      
      // Load detection model
      final detectionFile = '${appDir.path}/assets/models/rep_fast_base.onnx';
      final rawDetectionFile = await rootBundle.load(detectionFile);
      final detectionBytes = rawDetectionFile.buffer.asUint8List();
      detectionModel = OrtSession.fromBuffer(detectionBytes, sessionOptions);
      
      // Load recognition model
      final recognitionFile = '${appDir.path}/assets/models/crnn_mobilenet_v3_large.onnx';
      final rawRecognitionFile = await rootBundle.load(recognitionFile);
      final recognitionBytes = rawRecognitionFile.buffer.asUint8List();
      recognitionModel = OrtSession.fromBuffer(recognitionBytes, sessionOptions);
      
    } catch (e) {
      throw Exception('Error loading models: $e');
    }
}

  Future<img_lib.Image?> uiImageToImage(ui.Image image) async {
    try {
      final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) return null;

      return img_lib.Image.fromBytes(
        width: image.width,
        height: image.height,
        bytes: byteData.buffer,
        order: img_lib.ChannelOrder.rgba,
      );
    } catch (e) {
      throw Exception('Error converting UI Image: $e');
    }
  }

Future<Float32List> preprocessImageForDetection(ui.Image image) async {
  final img = await uiImageToImage(image);
  if (img == null) throw Exception('Failed to process image');

  final resized = img_lib.copyResize(
    img,
    width: OCRConstants.TARGET_SIZE[0],
    height: OCRConstants.TARGET_SIZE[1],
  );

  final preprocessedData = Float32List(OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1] * 3);
  
  for (int y = 0; y < resized.height; y++) {
    for (int x = 0; x < resized.width; x++) {
      final pixel = resized.getPixel(x, y);
      final idx = y * resized.width + x;
      
      // Use the r, g, b getters directly from the Pixel class
      preprocessedData[idx] = 
          (pixel.r.toDouble() / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0];
      preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1]] = 
          (pixel.g.toDouble() / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1];
      preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1] * 2] = 
          (pixel.b.toDouble() / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2];
    }
  }
  
  return preprocessedData;
}

Future<Map<String, dynamic>> detectText(ui.Image image) async {
    if (detectionModel == null) throw Exception('Detection model not loaded');
    
    try {
      final inputTensor = await preprocessImageForDetection(image);
      
      final tensor = OrtValueTensor.createTensorWithDataList(
        inputTensor,
        [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]]
      );

      // Create input feeds map
      final Map<String, OrtValue> inputs = {'input': tensor};
      
      // Create run options
      final runOptions = OrtRunOptions();

      List<OrtValue?>? outputs;

      try {
        // Run model inference using runAsync
        outputs = await detectionModel?.runAsync(runOptions, inputs);
        
        if (outputs == null || outputs.isEmpty) {
          throw Exception('No output from detection model');
        }

        final probMap = outputs[0]?.value as Float32List;
        if (probMap == null) {
          throw Exception('Invalid output tensor');
        }
        
        final processedProbMap = Float32List.fromList(
          probMap.map((x) => 1.0 / (1.0 + math.exp(-x))).toList()
        );

        return {
          'out_map': processedProbMap,
          'preds': postprocessProbabilityMap(processedProbMap),
        };
      } finally {
        // Cleanup resources
        tensor.release();
        runOptions.release();
        outputs?.forEach((element) {
          element?.release();
        });
      }
    } catch (e) {
      throw Exception('Error running detection model: $e');
    }
}



Future<List<BoundingBox>> extractBoundingBoxes(Float32List probMap) async {
    final imgWidth = OCRConstants.TARGET_SIZE[0];
    final imgHeight = OCRConstants.TARGET_SIZE[1];
    
    try {
      // Convert probability map to grayscale image
      final Uint8List grayImage = Uint8List(imgWidth * imgHeight);
      for (int i = 0; i < probMap.length; i++) {
        grayImage[i] = (probMap[i] * 255).round().clamp(0, 255);
      }

      // Create image from grayscale data
      final mat = img_lib.Image.fromBytes(
        width: imgWidth,
        height: imgHeight,
        bytes: grayImage.buffer,
        numChannels: 1,
        order: img_lib.ChannelOrder.rgba,
      );

      // Apply threshold
      final binaryImage = img_lib.Image(
        width: imgWidth,
        height: imgHeight,
      );
    
      for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
          final pixelValue = mat.getPixel(x, y).r > 77 ? 255 : 0;
          binaryImage.setPixel(x, y, img_lib.ColorRgb8(pixelValue, pixelValue, pixelValue));
        }
      }

      // Find connected components
      List<BoundingBox> boundingBoxes = [];
      List<List<bool>> visited = List.generate(
        imgHeight,
        (_) => List.filled(imgWidth, false),
      );

      for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
          if (!visited[y][x] && binaryImage.getPixel(x, y).r == 255) {
            var bounds = _findComponentBounds(binaryImage, visited, x, y);
            
            final width = bounds['maxX']! - bounds['minX']! + 1;
            final height = bounds['maxY']! - bounds['minY']! + 1;
            
            if (width > 2 && height > 2) {
              final box = _transformBoundingBox(
                bounds['minX']!.toDouble(),
                bounds['minY']!.toDouble(),
                width.toDouble(),
                height.toDouble(),
                imgWidth.toDouble(),
                imgHeight.toDouble(),
              );
              boundingBoxes.add(box);
            }
          }
        }
      }

      return boundingBoxes;
    } catch (e) {
      throw Exception('Error extracting bounding boxes: $e');
    }
  }

Map<String, int> _findComponentBounds(
    img_lib.Image image,
    List<List<bool>> visited,
    int startX,
    int startY,
  ) {
    int minX = startX, maxX = startX, minY = startY, maxY = startY;
    final queue = <List<int>>[];
    queue.add([startX, startY]);
    
    while (queue.isNotEmpty) {
      final point = queue.removeAt(0);
      final x = point[0], y = point[1];
      
      if (x < 0 || x >= image.width || y < 0 || y >= image.height ||
          visited[y][x] || image.getPixel(x, y).r != 255) {
        continue;
      }
      
      visited[y][x] = true;
      minX = math.min(minX, x);
      maxX = math.max(maxX, x);
      minY = math.min(minY, y);
      maxY = math.max(maxY, y);
      
      queue.add([x + 1, y]);
      queue.add([x - 1, y]);
      queue.add([x, y + 1]);
      queue.add([x, y - 1]);
    }
    
    return {
      'minX': minX,
      'maxX': maxX,
      'minY': minY,
      'maxY': maxY,
    };
  }  

void _floodFill(
  img_lib.Image image,
  List<List<bool>> visited,
  int x,
  int y,
  _BoundingRegion region,
) {
  if (x < 0 || x >= image.width || 
      y < 0 || y >= image.height ||
      visited[y][x] || 
      image.getPixel(x, y) != 255) {
    return;
  }

  visited[y][x] = true;
  region.update(x, y);

  // Check all four directions
  _floodFill(image, visited, x + 1, y, region);
  _floodFill(image, visited, x - 1, y, region);
  _floodFill(image, visited, x, y + 1, region);
  _floodFill(image, visited, x, y - 1, region);
}

  List<int> postprocessProbabilityMap(Float32List probMap) {
    const threshold = 0.1;
    return probMap.map((prob) => prob > threshold ? 1 : 0).toList();
  }

Future<Float32List> preprocessImageForRecognition(List<ui.Image> crops) async {
  final processedImages = await Future.wait(crops.map((crop) async {
    final img_lib.Image? processed = await uiImageToImage(crop);
    if (processed == null) throw Exception('Failed to process crop');

    final resized = img_lib.copyResize(
      processed,
      width: OCRConstants.RECOGNITION_TARGET_SIZE[1],
      height: OCRConstants.RECOGNITION_TARGET_SIZE[0],
    );

    final Float32List float32Data = Float32List(3 * OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]);
    
    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        final pixel = resized.getPixel(x, y);
        final idx = y * resized.width + x;
        
        // Use the r, g, b getters directly from the Pixel class
        float32Data[idx] = 
            (pixel.r.toDouble() / 255.0 - OCRConstants.REC_MEAN[0]) / OCRConstants.REC_STD[0];
        float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]] = 
            (pixel.g.toDouble() / 255.0 - OCRConstants.REC_MEAN[1]) / OCRConstants.REC_STD[1];
        float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1] * 2] = 
            (pixel.b.toDouble() / 255.0 - OCRConstants.REC_MEAN[2]) / OCRConstants.REC_STD[2];
      }
    }
    
    return float32Data;
  }));

  // Combine all processed images
  final combinedData = Float32List(3 * OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1] * processedImages.length);
  for (int i = 0; i < processedImages.length; i++) {
    combinedData.setAll(i * processedImages[0].length, processedImages[i]);
  }
  
  return combinedData;
}

Future<Map<String, dynamic>> recognizeText(List<ui.Image> crops) async {
    if (recognitionModel == null) throw Exception('Recognition model not loaded');
    
    try {
      final preprocessedData = await preprocessImageForRecognition(crops);
      
      final tensor = OrtValueTensor.createTensorWithDataList(
        preprocessedData,
        [crops.length, 3, OCRConstants.RECOGNITION_TARGET_SIZE[0], OCRConstants.RECOGNITION_TARGET_SIZE[1]]
      );

      // Create input feeds map
      final Map<String, OrtValue> inputs = {'input': tensor};
      
      // Create run options
      final runOptions = OrtRunOptions();
      
      List<OrtValue?>? outputs;

      try {
        // Run model inference using runAsync
         outputs = await recognitionModel?.runAsync(runOptions, inputs);
        
        if (outputs == null || outputs.isEmpty) {
          throw Exception('No output from recognition model');
        }

        final logits = outputs[0]?.value as Float32List;
        if (logits == null) {
          throw Exception('Invalid output tensor');
        }

        final batchSize = crops.length;
        final height = OCRConstants.RECOGNITION_TARGET_SIZE[0];
        final numClasses = OCRConstants.VOCAB.length + 1; // +1 for blank token

        // Process logits and apply softmax
        final probabilities = List.generate(batchSize, (b) {
          return List.generate(height, (h) {
            final positionLogits = List.generate(numClasses, (c) {
              final index = b * (height * numClasses) + h * numClasses + c;
              return logits[index].toDouble();
            });
            return _softmax(positionLogits);
          });
        });

        // Find best path and decode text
        final bestPath = probabilities.map((batchProb) {
          return batchProb.map((row) {
            return row.indexOf(row.reduce(math.max));
          }).toList();
        }).toList();

        final decodedTexts = bestPath.map((sequence) {
          return sequence
              .where((idx) => idx != numClasses - 1)
              .map((idx) => OCRConstants.VOCAB[idx])
              .join('');
        }).toList();

        return {
          'probabilities': probabilities,
          'bestPath': bestPath,
          'decodedTexts': decodedTexts,
        };
      } finally {
        // Cleanup resources
        tensor.release();
        runOptions.release();
        outputs?.forEach((element) {
          element?.release();
        });
      }
    } catch (e) {
      throw Exception('Error running recognition model: $e');
    }
}

// Helper function to convert Float32List values to double
double _convertToDouble(dynamic value) {
  if (value is double) return value;
  if (value is int) return value.toDouble();
  if (value is num) return value.toDouble();
  throw Exception('Cannot convert value to double: $value');
}



List<int> _extractRGB(img_lib.Pixel pixel) {
  // Use the correct method to extract RGB values from image library Pixel
  return [
    pixel.r.toInt(),  // Red component
    pixel.g.toInt(),  // Green component
    pixel.b.toInt(),  // Blue component
  ];
}

Future<List<ui.Image>> cropImages(ui.Image sourceImage, List<BoundingBox> boxes) async {
  List<ui.Image> crops = [];
  
  for (final box in boxes) {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    
    // Create source and destination rectangles
    final srcRect = Rect.fromLTWH(box.x, box.y, box.width, box.height);
    final dstRect = Rect.fromLTWH(0, 0, box.width, box.height);
    
    // Draw the cropped portion
    canvas.drawImageRect(sourceImage, srcRect, dstRect, Paint());
    
    // Convert to image
    final picture = recorder.endRecording();
    final croppedImage = await picture.toImage(
      box.width.round(),
      box.height.round(),
    );
    
    crops.add(croppedImage);
  }
  
  return crops;
}

List<double> _softmax(List<double> input) {
  final maxVal = input.reduce(math.max);
  final expValues = input.map((x) => math.exp(x - maxVal)).toList();
  final sumExp = expValues.reduce((a, b) => a + b);
  return expValues.map((x) => x / sumExp).toList();
}

String _ctcDecode(List<int> sequence, int blankIndex) {
  final result = StringBuffer();
  int? previousClass;
  
  for (final currentClass in sequence) {
    if (currentClass != blankIndex && currentClass != previousClass) {
      if (currentClass < OCRConstants.VOCAB.length) {
        result.write(OCRConstants.VOCAB[currentClass]);
      }
    }
    previousClass = currentClass;
  }
  
  return result.toString();
}

BoundingBox _transformBoundingBox(
  double x,
  double y,
  double width,
  double height,
  double imgWidth,
  double imgHeight,
) {
  // Calculate padding based on box dimensions
  final offset = (width * height * 1.8) / (2 * (width + height));
  
  // Apply padding and ensure coordinates stay within image bounds
  final x1 = _clamp(x - offset, 0, imgWidth);
  final y1 = _clamp(y - offset, 0, imgHeight);
  
  // Calculate end coordinates with padding
  final x2 = _clamp(x + width + offset, 0, imgWidth);
  final y2 = _clamp(y + height + offset, 0, imgHeight);
  
  return BoundingBox(
    x: x1,
    y: y1,
    width: x2 - x1,
    height: y2 - y1,
  );
}

Future<List<OCRResult>> processImage(ui.Image image) async {
  try {
    // Step 1: Detection
    final detectionResult = await detectText(image);
    if (detectionResult.isEmpty) {
      return [];
    }
    
    // Step 2: Extract bounding boxes
    final boundingBoxes = await extractBoundingBoxes(detectionResult['out_map'] as Float32List);
    if (boundingBoxes.isEmpty) {
      return [];
    }
    
    // Step 3: Crop images based on bounding boxes
    final crops = await cropImages(image, boundingBoxes);
    
    // Step 4: Recognition
    final recognitionResult = await recognizeText(crops);
    final decodedTexts = recognitionResult['decodedTexts'] as List<String>;
    
    // Step 5: Combine results
    return List.generate(
      decodedTexts.length,
      (i) => OCRResult(
        text: decodedTexts[i],
        boundingBox: boundingBoxes[i],
      ),
    ).where((result) => result.text.isNotEmpty).toList();
    
  } catch (e) {
    print('Error processing image: $e');
    return [];
  }
}

// Helper method for clamping values
double _clamp(double value, double min, double max) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

}