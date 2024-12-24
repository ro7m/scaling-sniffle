import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' show exp, max;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img_lib;
import 'package:path_provider/path_provider.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class OCRResult {
  final String text;
  final BoundingBox boundingBox;

  OCRResult({required this.text, required this.boundingBox});
}


class OCRService {
  OrtSession? detectionModel;
  OrtSession? recognitionModel;
  
  Future<void> loadModels() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      
      // Initialize ONNX Runtime
      final options = OrtSessionOptions();
      options.setIntraOpNumThreads(1);
      options.setInterOpNumThreads(1);
      
      // Load detection model
      final detectionFile = File('${appDir.path}/assets/models/rep_fast_base.onnx');
      detectionModel = await OrtSession.fromFile(detectionFile, options);
      
      // Load recognition model
      final recognitionFile = File('${appDir.path}/assets/models/crnn_mobilenet_v3_large.onnx');
      recognitionModel = await OrtSession.fromFile(recognitionFile, options);
      
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
        numChannels: 4,
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
        final color = img_lib.getColor(resized, x, y);
        final idx = y * resized.width + x;
        
        preprocessedData[idx] = 
            (color.r / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0];
        preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1]] = 
            (color.g / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1];
        preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1] * 2] = 
            (color.b / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2];
      }
    }
    
    return preprocessedData;
  }

    Future<Map<String, dynamic>> detectText(ui.Image image) async {
    if (detectionModel == null) throw Exception('Detection model not loaded');
    
    try {
      final inputTensor = await preprocessImageForDetection(image);
      
      // Create ONNX tensor
      final tensor = OrtTensor.fromList(
        TensorElementType.float,
        inputTensor,
        [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]]
      );

      final feeds = {'input': tensor};
      final results = await detectionModel!.run(feeds);
      final probMap = results.values.first.value as Float32List;
      
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
      // Convert to OpenCV matrix
      final matData = probMap.map((x) => (x * 255).toInt().clamp(0, 255)).toList();
      final mat = cv.Mat.create(imgHeight, imgWidth, cv.CV_8UC1);
      mat.data = matData;

      // Apply threshold
      final binary = cv.Mat.create(imgHeight, imgWidth, cv.CV_8UC1);
      cv.threshold(mat, binary, 77, 255, cv.THRESH_BINARY);
      
      // Find contours
      final List<List<cv.Point>> contours = [];
      final hierarchy = cv.Mat.create(1, 1, cv.CV_32SC4);
      cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      final boundingBoxes = <BoundingBox>[];
      
      for (final contour in contours) {
        final rect = cv.boundingRect(contour);
        if (rect.width > 2 && rect.height > 2) {
          boundingBoxes.add(BoundingBox(
            x: rect.x.toDouble(),
            y: rect.y.toDouble(),
            width: rect.width.toDouble(),
            height: rect.height.toDouble(),
          ));
        }
      }

      mat.release();
      binary.release();
      hierarchy.release();

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
          
          float32Data[idx] = 
              (img_lib.getRed(pixel) / 255.0 - OCRConstants.REC_MEAN[0]) / OCRConstants.REC_STD[0];
          float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]] = 
              (img_lib.getGreen(pixel) / 255.0 - OCRConstants.REC_MEAN[1]) / OCRConstants.REC_STD[1];
          float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1] * 2] = 
              (img_lib.getBlue(pixel) / 255.0 - OCRConstants.REC_MEAN[2]) / OCRConstants.REC_STD[2];
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
      
      final feeds = {
        'input': OrtValueTensor.createTensor(
          preprocessedData,
          [crops.length, 3, OCRConstants.RECOGNITION_TARGET_SIZE[0], OCRConstants.RECOGNITION_TARGET_SIZE[1]],
        )
      };

      final results = await recognitionModel!.run(feeds);
      final logits = results['logits']!.data as Float32List;
      final dims = results['logits']!.shape;
      
      final batchSize = dims[0];
      final height = dims[1];
      final numClasses = dims[2];

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

      // Find best path
      final bestPath = probabilities.map((batchProb) {
        return batchProb.map((row) {
          return row.indexOf(row.reduce(max));
        }).toList();
      }).toList();

      // Decode text
      final decodedTexts = bestPath.map((sequence) {
        return sequence
            .where((idx) => idx != numClasses - 1) // Remove blank token
            .map((idx) => OCRConstants.VOCAB[idx])
            .join('');
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

  List<double> _softmax(List<double> input) {
    final maxVal = input.reduce(max);
    final exp_values = input.map((x) => exp(x - maxVal)).toList();
    final sumExp = exp_values.reduce((a, b) => a + b);
    return exp_values.map((x) => x / sumExp).toList();
  }

  Future<List<ui.Image>> cropImages(ui.Image sourceImage, List<BoundingBox> boxes) async {
    List<ui.Image> crops = [];
    
    for (final box in boxes) {
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      
      canvas.drawImageRect(
        sourceImage,
        Rect.fromLTWH(box.x, box.y, box.width, box.height),
        Rect.fromLTWH(0, 0, box.width, box.height),
        Paint(),
      );
      
      final picture = recorder.endRecording();
      final croppedImage = await picture.toImage(
        box.width.round(),
        box.height.round(),
      );
      
      crops.add(croppedImage);
    }
    
    return crops;
  }

  BoundingBox _transformBoundingBox(
    double x,
    double y,
    double width,
    double height,
    double imgWidth,
    double imgHeight,
  ) {
    final offset = (width * height * 1.8) / (2 * (width + height));
    
    final x1 = _clamp(x - offset, imgWidth);
    final x2 = _clamp(x1 + width + 2 * offset, imgWidth);
    final y1 = _clamp(y - offset, imgHeight);
    final y2 = _clamp(y1 + height + 2 * offset, imgHeight);
    
    return BoundingBox(
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1,
    );
  }

  double _clamp(double value, double max) {
    return value.clamp(0, max);
  }

  Future<List<OCRResult>> processImage(ui.Image image) async {
    try {
      // Step 1: Detection
      final detectionResult = await detectText(image);
      
      // Step 2: Extract bounding boxes
      final boundingBoxes = await extractBoundingBoxes(detectionResult['out_map']);
      
      // Step 3: Crop images based on bounding boxes
      final crops = await cropImages(image, boundingBoxes);
      
      // Step 4: Recognition
      final recognitionResult = await recognizeText(crops);
      
      // Step 5: Combine results
      List<OCRResult> results = [];
      for (int i = 0; i < recognitionResult['decodedTexts'].length; i++) {
        results.add(OCRResult(
          text: recognitionResult['decodedTexts'][i],
          boundingBox: boundingBoxes[i],
        ));
      }
      
      return results;
    } catch (e) {
      throw Exception('Error processing image: $e');
    }
  }
}