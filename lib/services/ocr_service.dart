import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:flutter_onnx_runtime/flutter_onnx_runtime.dart';
import '../constants.dart';
import '../models/bounding_box.dart';

class OCRService {
  late OrtSession detectionModel;
  late OrtSession recognitionModel;
  
  Future<void> loadModels() async {
    final appDir = await getApplicationDocumentsDirectory();
    
    // Load detection model
    final detectionPath = '${appDir.path}/models/rep_fast_base.onnx';
    detectionModel = await OrtSession.fromFile(detectionPath);
    
    // Load recognition model
    final recognitionPath = '${appDir.path}/models/crnn_mobilenet_v3_large.onnx';
    recognitionModel = await OrtSession.fromFile(recognitionPath);
  }

  Float32List preprocessImageForDetection(ui.Image image) {
    // Convert image to bytes
    final img.Image? processedImage = img.copyResize(
      img.Image.fromBytes(
        image.width,
        image.height,
        image.toByteData()!.buffer.asUint8List(),
        channels: 4,
      ),
      width: OCRConstants.TARGET_SIZE[0],
      height: OCRConstants.TARGET_SIZE[1],
    );

    if (processedImage == null) throw Exception('Failed to process image');

    final Float32List preprocessedData = Float32List(OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1] * 3);
    
    for (int y = 0; y < processedImage.height; y++) {
      for (int x = 0; x < processedImage.width; x++) {
        final pixel = processedImage.getPixel(x, y);
        final int idx = y * processedImage.width + x;
        
        // Normalize and apply mean/std
        preprocessedData[idx] = 
            (img.getRed(pixel) / 255.0 - OCRConstants.DET_MEAN[0]) / OCRConstants.DET_STD[0];
        preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1]] = 
            (img.getGreen(pixel) / 255.0 - OCRConstants.DET_MEAN[1]) / OCRConstants.DET_STD[1];
        preprocessedData[idx + OCRConstants.TARGET_SIZE[0] * OCRConstants.TARGET_SIZE[1] * 2] = 
            (img.getBlue(pixel) / 255.0 - OCRConstants.DET_MEAN[2]) / OCRConstants.DET_STD[2];
      }
    }
    
    return preprocessedData;
  }

  Future<Map<String, dynamic>> detectText(ui.Image image) async {
    final inputTensor = preprocessImageForDetection(image);
    
    final feeds = {
      'input': OrtValueTensor.fromList(
        inputTensor,
        [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]],
      )
    };

    final results = await detectionModel.run(feeds);
    final probMap = results.values.first.data as Float32List;
    
    // Apply sigmoid
    final processedProbMap = Float32List.fromList(
      probMap.map((x) => 1.0 / (1.0 + exp(-x))).toList()
    );

    return {
      'out_map': processedProbMap,
      'preds': postprocessProbabilityMap(processedProbMap),
    };
  }

  List<int> postprocessProbabilityMap(Float32List probMap) {
    const threshold = 0.1;
    return probMap.map((prob) => prob > threshold ? 1 : 0).toList();
  }

  Float32List preprocessImageForRecognition(List<ui.Image> crops) {
    final processedImages = crops.map((crop) {
      final img.Image resized = img.copyResize(
        img.Image.fromBytes(
          crop.width,
          crop.height,
          crop.toByteData()!.buffer.asUint8List(),
          channels: 4,
        ),
        width: OCRConstants.RECOGNITION_TARGET_SIZE[1],
        height: OCRConstants.RECOGNITION_TARGET_SIZE[0],
      );

      final Float32List float32Data = Float32List(3 * OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]);
      
      for (int y = 0; y < resized.height; y++) {
        for (int x = 0; x < resized.width; x++) {
          final pixel = resized.getPixel(x, y);
          final idx = y * resized.width + x;
          
          float32Data[idx] = 
              (img.getRed(pixel) / 255.0 - OCRConstants.REC_MEAN[0]) / OCRConstants.REC_STD[0];
          float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1]] = 
              (img.getGreen(pixel) / 255.0 - OCRConstants.REC_MEAN[1]) / OCRConstants.REC_STD[1];
          float32Data[idx + OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1] * 2] = 
              (img.getBlue(pixel) / 255.0 - OCRConstants.REC_MEAN[2]) / OCRConstants.REC_STD[2];
        }
      }
      
      return float32Data;
    }).toList();

    // Combine all processed images
    final combinedData = Float32List(3 * OCRConstants.RECOGNITION_TARGET_SIZE[0] * OCRConstants.RECOGNITION_TARGET_SIZE[1] * processedImages.length);
    for (int i = 0; i < processedImages.length; i++) {
      combinedData.setAll(i * processedImages[0].length, processedImages[i]);
    }
    
    return combinedData;
  }

  Future<Map<String, dynamic>> recognizeText(List<ui.Image> crops) async {
    final preprocessedData = preprocessImageForRecognition(crops);
    
    final feeds = {
      'input': OrtValueTensor.fromList(
        preprocessedData,
        [crops.length, 3, OCRConstants.RECOGNITION_TARGET_SIZE[0], OCRConstants.RECOGNITION_TARGET_SIZE[1]],
      )
    };

    final results = await recognitionModel.run(feeds);
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
          return logits[index];
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
  }

  List<double> _softmax(List<double> input) {
    final maxVal = input.reduce(max);
    final exp = input.map((x) => exp(x - maxVal)).toList();
    final sumExp = exp.reduce((a, b) => a + b);
    return exp.map((x) => x / sumExp).toList();
  }

  Future<List<BoundingBox>> extractBoundingBoxes(Float32List probMap) async {
    // Convert probability map to image for OpenCV processing
    final imgWidth = OCRConstants.TARGET_SIZE[0];
    final imgHeight = OCRConstants.TARGET_SIZE[1];
    
    final byteData = Float32List.fromList(probMap.map((x) => x * 255).toList())
        .buffer.asUint8List();
    
    // Use OpenCV to find contours
    final matrix = await ImgProc.threshold(
      byteData,
      imgWidth,
      imgHeight,
      77,
      255,
      ImgProc.threshBinary,
    );

    // Apply morphological operation
    final kernel = await ImgProc.getStructuringElement(
      ImgProc.morphRect,
      [2, 2],
    );
    final opened = await ImgProc.morphologyEx(
      matrix,
      ImgProc.morphOpen,
      kernel,
      kernelSize: [2, 2],
    );

    // Find contours
    final contours = await ImgProc.findContours(
      opened,
      ImgProc.retrExternal,
      ImgProc.chainApproxSimple,
    );

    // Process contours to get bounding boxes
    List<BoundingBox> boundingBoxes = [];
    for (final contour in contours) {
      final rect = await ImgProc.boundingRect(contour);
      
      if (rect[2] > 2 && rect[3] > 2) { // width & height > 2
        final box = _transformBoundingBox(
          rect[0].toDouble(), // x
          rect[1].toDouble(), // y
          rect[2].toDouble(), // width
          rect[3].toDouble(), // height
          imgWidth.toDouble(),
          imgHeight.toDouble(),
        );
        boundingBoxes.add(box);
      }
    }

    return boundingBoxes;
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

  Future<List<ui.Image>> cropImages(ui.Image sourceImage, List<BoundingBox> boxes) async {
    List<ui.Image> crops = [];
    
    for (final box in boxes) {
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      
      // Draw the cropped portion
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

  Future<List<OCRResult>> processImage(ui.Image image) async {
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
  }

}