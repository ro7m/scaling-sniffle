import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' show exp, max;
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

class OCRService {
  OrtSession? detectionModel;
  OrtSession? recognitionModel;
  
  Future<void> loadModels() async {
    try {
      final sessionOptions = OrtSessionOptions();
      final appDir = await getApplicationDocumentsDirectory();
      
      // Load detection model
      final detectionFile = '${appDir.path}/assets/models/rep_fast_base.onnx';
      final rawdetectionFile = await rootBundle.load(detectionFile);
      final detectionBytes = rawdetectionFile.buffer.asUint8List();
      detectionModel = await OrtSession.fromBuffer(detectionBytes, sessionOptions);
      
      // Load recognition model
      final recognitionFile = '${appDir.path}/assets/models/crnn_mobilenet_v3_large.onnx';
      final rawrecognitionFile = await rootBundle.load(recognitionFile);
      final recognitionBytes = rawrecognitionFile.buffer.asUint8List();
      recognitionModel = await OrtSession.fromBuffer(recognitionBytes, sessionOptions);
      
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
        final r = (pixel >> 16) & 0xFF;
        final g = (pixel >> 8) & 0xFF;
        final b = pixel & 0xFF;
        final idx = y * resized.width + x;
        
        preprocessedData[idx] = 
            (r / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0];
        preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1]] = 
            (g / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1];
        preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1] * 2] = 
            (b / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2];
      }
    }
    
    return preprocessedData;
  }

  Future<Map<String, dynamic>> detectText(ui.Image image) async {
    if (detectionModel == null) throw Exception('Detection model not loaded');
    
    try {
      final inputTensor = await preprocessImageForDetection(image);
      
      final tensor = OrtValueTensor.createTensorWithDataList(
        inputTensor.toList(),
        [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]],
      );

      final feeds = {'input': tensor};
      final outputs = {'output': OrtValueTensor};
      final results = await detectionModel!.run(feeds, outputs);
      final probMap = results['output']!.value as Float32List;
      
      final processedProbMap = Float32List.fromList(
        probMap.map((x) => 1.0 / (1.0 + exp(-x))).toList()
      );

      return {
        'out_map': processedProbMap,
        'preds': postprocessProbabilityMap(processedProbMap),
      };
    } catch (e) {
      throw Exception('Error running detection model: $e');
    }
  }

  Future<List<BoundingBox>> extractBoundingBoxes(Float32List probMap) async {
    final imgWidth = OCRConstants.TARGET_SIZE[0];
    final imgHeight = OCRConstants.TARGET_SIZE[1];
    
    try {
      final mat = cv.Mat.fromArray(
        probMap.map((x) => (x * 255).toInt().clamp(0, 255)).toList(),
        cv.CV_8UC1,
        imgHeight,
        imgWidth,
      );
      
      final threshold = cv.Mat();
      cv.threshold(mat, threshold, 77, 255, cv.THRESH_BINARY);

      final kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        cv.Size(2, 2),
      );
      
      final opened = cv.Mat();
      cv.morphologyEx(threshold, opened, cv.MORPH_OPEN, kernel);

      final contours = cv.findContours(
        opened,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
      );

      List<BoundingBox> boundingBoxes = [];
      
      for (final contour in contours) {
        try {
          final rect = cv.boundingRect(contour);
          
          if (rect.width > 2 && rect.height > 2) {
            final box = _transformBoundingBox(
              rect.x.toDouble(),
              rect.y.toDouble(),
              rect.width.toDouble(),
              rect.height.toDouble(),
              imgWidth.toDouble(),
              imgHeight.toDouble(),
            );
            boundingBoxes.add(box);
          }
        } catch (e) {
          print('Error processing contour: $e');
          continue;
        }
      }

      // Clean up OpenCV resources
      mat.release();
      threshold.release();
      kernel.release();
      opened.release();

      return boundingBoxes;
    } catch (e) {
      throw Exception('Error extracting bounding boxes: $e');
    }
  }

  List<int> postprocessProbabilityMap(Float32List probMap) {
    const threshold = 0.1;
    return probMap.map((prob) => prob > threshold ? 1 : 0).toList();
  }

  Future<Float32List> preprocessImageForRecognition(List<ui.Image> crops) async {
    final processedImages = await Future.wait(crops.map((crop) async {
      final processed = await uiImageToImage(crop);
      if (processed == null) throw Exception('Failed to process crop');

      final resized = img_lib.copyResize(
        processed,
        width: OCRConstants.RECOGNITION_TARGET_SIZE[1],
        height: OCRConstants.RECOGNITION_TARGET_SIZE[0],
      );

      final float32Data = Float32List(3 * OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]);
      
      for (int y = 0; y < resized.height; y++) {
        for (int x = 0; x < resized.width; x++) {
          final pixel = resized.getPixel(x, y);
          final r = (pixel >> 16) & 0xFF;
          final g = (pixel >> 8) & 0xFF;
          final b = pixel & 0xFF;
          final idx = y * resized.width + x;
          
          float32Data[idx] = 
              (r / 255.0 - OCRConstants.REC_MEAN[0]) / OCRConstants.REC_STD[0];
          float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]] = 
              (g / 255.0 - OCRConstants.REC_MEAN[1]) / OCRConstants.REC_STD[1];
          float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1] * 2] = 
              (b / 255.0 - OCRConstants.REC_MEAN[2]) / OCRConstants.REC_STD[2];
        }
      }
      
      return float32Data;
    }));

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
    
    // Create input tensor
    final tensor = OrtValueTensor.createTensorWithDataList(
      preprocessedData,
      [crops.length, 3, OCRConstants.RECOGNITION_TARGET_SIZE[0], OCRConstants.RECOGNITION_TARGET_SIZE[1]]
    );

    // Create input feeds
    final Map<String, OrtValue> feeds = {'input': tensor};
    
    // Create output map
    final Map<String, OrtValue> outputs = {};
    
    // Run model with proper OrtRunOptions
    final runOptions = OrtRunOptions();
    await recognitionModel!.run(runOptions, feeds, outputs);
    
    // Get logits from output
    final logits = outputs['logits']?.value as Float32List;
    if (logits == null) throw Exception('No output logits found');
    
    // Get tensor dimensions
    final tensorInfo = outputs['logits']?.typeInfo as OrtTensorTypeInfo;
    final dims = tensorInfo.shape;
    
    final batchSize = dims[0];
    final height = dims[1];
    final numClasses = dims[2];

    // Process logits and apply softmax
    final probabilities = List.generate(batchSize, (b) {
      return List.generate(height, (h) {
        final positionLogits = List.generate(numClasses, (c) {
          final idx = b * (height * numClasses) + h * numClasses + c;
          return logits[idx];
        }).map((x) => x.toDouble()).toList();
        return _softmax(positionLogits);
      });
    });

    // Find best path
    final bestPath = probabilities.map((batchProb) {
      return batchProb.map((row) {
        return row.indexOf(row.reduce(max));
      }).toList();
    }).toList();

    // Decode text
    final decodedTexts = bestPath.map((sequence) {
      return _ctcDecode(sequence, numClasses - 1);
    }).toList();

    return {
      'probabilities': probabilities,
      'bestPath': bestPath,
      'decodedTexts': decodedTexts,
    };
  } catch (e) {
    throw Exception('Error running recognition model: $e');
  }
}

// Helper function to extract RGB values from image pixel
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
  final maxVal = input.reduce(max);
  final expValues = input.map((x) => exp(x - maxVal)).toList();
  final sumExp = expValues.reduce((a, b) => a + b);
  return expValues.map((x) => x / sumExp).toList();
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