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
import '../constants.dart';
import '../models/bounding_box.dart';
import '../models/ocr_result.dart';

class OCRService {
  OrtSession? detectionModel;
  OrtSession? recognitionModel;
  void Function(String)? debugCallback;

  // Constructor to accept debug callback
  OCRService({this.debugCallback});

  Future<void> loadModels() async {
    try {
      debugCallback?.call('Starting to load models...');
      final sessionOptions = OrtSessionOptions();
      
      // Load detection model
      debugCallback?.call('Loading detection model...');
      final detectionBytes = await rootBundle.load('assets/models/rep_fast_base.onnx');
      debugCallback?.call('Detection model loaded from assets, size: ${detectionBytes.lengthInBytes / (1024 * 1024)} MB');
      
      detectionModel = OrtSession.fromBuffer(
        detectionBytes.buffer.asUint8List(
          detectionBytes.offsetInBytes,
          detectionBytes.lengthInBytes
        ),
        sessionOptions
      );
      debugCallback?.call('Detection model initialized successfully');
      
      // Load recognition model
      debugCallback?.call('Loading recognition model...');
      final recognitionBytes = await rootBundle.load('assets/models/crnn_mobilenet_v3_large.onnx');
      debugCallback?.call('Recognition model loaded from assets, size: ${recognitionBytes.lengthInBytes / (1024 * 1024)} MB');
      
      recognitionModel = OrtSession.fromBuffer(
        recognitionBytes.buffer.asUint8List(
          recognitionBytes.offsetInBytes,
          recognitionBytes.lengthInBytes
        ),
        sessionOptions
      );
      debugCallback?.call('Recognition model initialized successfully');
      
    } catch (e, stackTrace) {
      debugCallback?.call('Error loading models: $e');
      debugCallback?.call('Stack trace: $stackTrace');
      throw Exception('Error loading models: $e');
    }
  }

  Future<Map<String, dynamic>> detectText(ui.Image image) async {
    if (detectionModel == null) {
      debugCallback?.call('Detection model not loaded');
      throw Exception('Detection model not loaded');
    }
    
    try {
      debugCallback?.call('Starting text detection...');
      debugCallback?.call('Preprocessing image for detection...');
      final inputTensor = await preprocessImageForDetection(image);
      debugCallback?.call('Image preprocessing completed');
      
      final tensor = OrtValueTensor.createTensorWithDataList(
        inputTensor,
        [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]]
      );
      debugCallback?.call('Input tensor created');

      final Map<String, OrtValue> inputs = {'input': tensor};
      final runOptions = OrtRunOptions();

      debugCallback?.call('Running detection model inference...');
      List<OrtValue?>? outputs;

      try {
        outputs = await detectionModel?.runAsync(runOptions, inputs);
        
        if (outputs == null || outputs.isEmpty) {
          debugCallback?.call('No output from detection model');
          throw Exception('No output from detection model');
        }

        final probMap = outputs[0]?.value as Float32List;
        if (probMap == null) {
          debugCallback?.call('Invalid output tensor');
          throw Exception('Invalid output tensor');
        }
        
        debugCallback?.call('Processing probability map...');
        final processedProbMap = Float32List.fromList(
          probMap.map((x) => 1.0 / (1.0 + math.exp(-x))).toList()
        );
        debugCallback?.call('Probability map processed');

        return {
          'out_map': processedProbMap,
          'preds': postprocessProbabilityMap(processedProbMap),
        };
      } finally {
        tensor.release();
        runOptions.release();
        outputs?.forEach((element) {
          element?.release();
        });
      }
    } catch (e, stackTrace) {
      debugCallback?.call('Error in detectText: $e');
      debugCallback?.call('Stack trace: $stackTrace');
      throw Exception('Error running detection model: $e');
    }
  }

  Future<Map<String, dynamic>> recognizeText(List<ui.Image> crops) async {
    if (recognitionModel == null) {
      debugCallback?.call('Recognition model not loaded');
      throw Exception('Recognition model not loaded');
    }
    
    try {
      debugCallback?.call('Starting text recognition for ${crops.length} crops...');
      debugCallback?.call('Preprocessing crops for recognition...');
      final preprocessedData = await preprocessImageForRecognition(crops);
      debugCallback?.call('Crops preprocessing completed');
      
      final tensor = OrtValueTensor.createTensorWithDataList(
        preprocessedData,
        [crops.length, 3, OCRConstants.RECOGNITION_TARGET_SIZE[0], OCRConstants.RECOGNITION_TARGET_SIZE[1]]
      );
      debugCallback?.call('Recognition input tensor created');

      final Map<String, OrtValue> inputs = {'input': tensor};
      final runOptions = OrtRunOptions();
      
      List<OrtValue?>? outputs;

      try {
        debugCallback?.call('Running recognition model inference...');
        outputs = await recognitionModel?.runAsync(runOptions, inputs);
        
        if (outputs == null || outputs.isEmpty) {
          debugCallback?.call('No output from recognition model');
          throw Exception('No output from recognition model');
        }

        final logits = outputs[0]?.value as Float32List;
        if (logits == null) {
          debugCallback?.call('Invalid recognition output tensor');
          throw Exception('Invalid output tensor');
        }

        debugCallback?.call('Processing recognition results...');
        final batchSize = crops.length;
        final height = OCRConstants.RECOGNITION_TARGET_SIZE[0];
        final numClasses = OCRConstants.VOCAB.length + 1;

        final probabilities = List.generate(batchSize, (b) {
          return List.generate(height, (h) {
            final positionLogits = List.generate(numClasses, (c) {
              final index = b * (height * numClasses) + h * numClasses + c;
              return logits[index].toDouble();
            });
            return _softmax(positionLogits);
          });
        });

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

        debugCallback?.call('Recognition completed. Found ${decodedTexts.length} text results');
        decodedTexts.forEach((text) {
          debugCallback?.call('Recognized text: "$text"');
        });

        return {
          'probabilities': probabilities,
          'bestPath': bestPath,
          'decodedTexts': decodedTexts,
        };
      } finally {
        tensor.release();
        runOptions.release();
        outputs?.forEach((element) {
          element?.release();
        });
      }
    } catch (e, stackTrace) {
      debugCallback?.call('Error in recognizeText: $e');
      debugCallback?.call('Stack trace: $stackTrace');
      throw Exception('Error running recognition model: $e');
    }
  }

  Future<List<OCRResult>> processImage(ui.Image image) async {
    try {
      debugCallback?.call('Starting image processing pipeline...');
      debugCallback?.call('Image dimensions: ${image.width}x${image.height}');

      // Step 1: Detection
      debugCallback?.call('Running text detection...');
      final detectionResult = await detectText(image);
      if (detectionResult.isEmpty) {
        debugCallback?.call('No text detected in the image');
        return [];
      }
      debugCallback?.call('Text detection completed');

      // Step 2: Extract bounding boxes
      debugCallback?.call('Extracting bounding boxes...');
      final boundingBoxes = await extractBoundingBoxes(detectionResult['out_map'] as Float32List);
      debugCallback?.call('Found ${boundingBoxes.length} bounding boxes');
      
      if (boundingBoxes.isEmpty) {
        debugCallback?.call('No bounding boxes found');
        return [];
      }

      // Debug bounding box information
      boundingBoxes.forEach((box) {
        debugCallback?.call('Box: (${box.x}, ${box.y}, ${box.width}, ${box.height})');
      });

      // Step 3: Crop images
      debugCallback?.call('Cropping images for recognition...');
      final crops = await cropImages(image, boundingBoxes);
      debugCallback?.call('Created ${crops.length} crops');

      // Step 4: Recognition
      debugCallback?.call('Starting text recognition...');
      final recognitionResult = await recognizeText(crops);
      final decodedTexts = recognitionResult['decodedTexts'] as List<String>;
      debugCallback?.call('Recognition completed');

      // Step 5: Combine results
      final results = List.generate(
        decodedTexts.length,
        (i) => OCRResult(
          text: decodedTexts[i],
          boundingBox: boundingBoxes[i],
        ),
      ).where((result) => result.text.isNotEmpty).toList();

      debugCallback?.call('Final results: ${results.length} valid text items found');
      results.forEach((result) {
        debugCallback?.call('Text: "${result.text}" at (${result.boundingBox.x}, ${result.boundingBox.y})');
      });

      return results;
    } catch (e, stackTrace) {
      debugCallback?.call('Error in processImage: $e');
      debugCallback?.call('Stack trace: $stackTrace');
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