import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import '../models/ocr_result.dart';
import '../models/bounding_box.dart';
import 'dart:math' as math;
import '../constants.dart'; 

// Helper class for bounding box calculation
class _BBox {
  int minX = 999999, minY = 999999;
  int maxX = -1, maxY = -1;
  
  void update(int x, int y) {
    minX = math.min(minX, x);
    minY = math.min(minY, y);
    maxX = math.max(maxX, x);
    maxY = math.max(maxY, y);
  }
  
  bool isValid() {
    return maxX >= 0 && maxY >= 0 && 
           maxX > minX && maxY > minY;
  }
}

class OCRService {
  OrtSession? detectionModel;
  OrtSession? recognitionModel;
  void Function(String)? debugCallback;

  // Load models
  Future<void> loadModels({void Function(String)? debugCallback}) async {
    this.debugCallback = debugCallback;
    try {
      debugCallback?.call('Starting to load models...');
      final sessionOptions = OrtSessionOptions();

      debugCallback?.call('Loading detection model...');
      final detectionBytes = await rootBundle.load('assets/models/rep_fast_base.onnx');
      detectionModel = await OrtSession.fromBuffer(
        detectionBytes.buffer.asUint8List(
          detectionBytes.offsetInBytes,
          detectionBytes.lengthInBytes
        ),
        sessionOptions
      );
      debugCallback?.call('Detection model loaded: ${detectionBytes.lengthInBytes ~/ 1024}KB');

      debugCallback?.call('Loading recognition model...');
      final recognitionBytes = await rootBundle.load('assets/models/crnn_mobilenet_v3_large.onnx');
      recognitionModel = await OrtSession.fromBuffer(
        recognitionBytes.buffer.asUint8List(
          recognitionBytes.offsetInBytes,
          recognitionBytes.lengthInBytes
        ),
        sessionOptions
      );
      debugCallback?.call('Recognition model loaded: ${recognitionBytes.lengthInBytes ~/ 1024}KB');
    } catch (e) {
      debugCallback?.call('Error loading models: $e');
      throw Exception('Failed to load models: $e');
    }
  }

  // Preprocess image for detection
  Future<Float32List> _preprocessImageForDetection(ui.Image image) async {
    debugCallback?.call('Preprocessing image for detection...');
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
    
    debugCallback?.call('Image preprocessing for detection completed');
    return preprocessedData;
  }

  // Run detection model
  Future<Float32List> _runDetection(Float32List preprocessedImage) async {
    try {
      final shape = [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]];
      final inputOrt = OrtValueTensor.createTensorWithDataList(preprocessedImage, shape);
      final inputs = {'input': inputOrt};
      final runOptions = OrtRunOptions();

      final results = await detectionModel?.runAsync(runOptions, inputs);
      inputOrt.release();
      runOptions.release();

      results?.forEach((element) {
        element?.release();
      });

      final output = results?.first.value;
      final List<dynamic> nestedOutput = output as List<dynamic>;

      return _convertToFloat32ListAndApplySigmoid(nestedOutput);
    } catch (e) {
      debugCallback?.call('Detection error: $e');
      throw Exception('Detection failed: $e');
    }
  }

  // Convert nested output to Float32List and apply sigmoid
  Float32List _convertToFloat32ListAndApplySigmoid(List<dynamic> nested) {
    final flattened = <double>[];
    
    void flatten(dynamic item) {
      if (item is List) {
        for (var subItem in item) {
          flatten(subItem);
        }
      } else if (item is num) {
        final sigmoid = 1.0 / (1.0 + math.exp(-item.toDouble()));
        flattened.add(sigmoid);
      }
    }
    
    flatten(nested);
    return Float32List.fromList(flattened);
  }

  // Extract bounding boxes
  Future<List<BoundingBox>> _extractBoundingBoxes(Float32List probMap) async {
    final List<BoundingBox> boxes = [];
    final threshold = 0.1;
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];
    
    List<List<bool>> binaryMap = List.generate(
      height,
      (_) => List.generate(width, (x) => probMap[x + _ * width] > threshold),
    );
    
    int currentLabel = 0;
    Map<int, _BBox> components = {};
    List<List<int>> labels = List.generate(height, (_) => List.filled(width, -1));
    
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (binaryMap[y][x] && labels[y][x] == -1) {
          _BBox bbox = _BBox();
          _floodFill(x, y, currentLabel, labels, binaryMap, bbox);
          
          if (bbox.isValid()) {
            components[currentLabel] = bbox;
            currentLabel++;
          }
        }
      }
    }
    
    for (var component in components.values) {
      final boxWidth = component.maxX - component.minX + 1;
      final boxHeight = component.maxY - component.minY + 1;
      
      if (boxWidth > 2 && boxHeight > 2) {
        double padding = (boxWidth * boxHeight * 1.8) / (2 * (boxWidth + boxHeight));
        
        double x1 = math.max(0, (component.minX - padding) / width);
        double y1 = math.max(0, (component.minY - padding) / height);
        double x2 = math.min(1.0, (component.maxX + padding) / width);
        double y2 = math.min(1.0, (component.maxY + padding) / height);
        
        boxes.add(BoundingBox(
          x: x1,
          y: y1,
          width: x2 - x1,
          height: y2 - y1,
        ));
      }
    }
    
    debugCallback?.call('Found ${boxes.length} bounding boxes');
    return boxes;
  }

  // Flood fill algorithm for connected components labeling
  void _floodFill(int x, int y, int label, List<List<int>> labels, List<List<bool>> binaryMap, _BBox bbox) {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];
    final queue = <math.Point<int>>[math.Point(x, y)];
    
    while (queue.isNotEmpty) {
      final point = queue.removeLast();
      final px = point.x;
      final py = point.y;
      
      if (px >= 0 && px < width && py >= 0 && py < height && binaryMap[py][px] && labels[py][px] == -1) {
        labels[py][px] = label;
        bbox.update(px, py);
        
        queue.add(math.Point(px + 1, py));
        queue.add(math.Point(px - 1, py));
        queue.add(math.Point(px, py + 1));
        queue.add(math.Point(px, py - 1));
      }
    }
  }

  // Preprocess image for recognition
  Future<Float32List> _preprocessImageForRecognition(ui.Image image) async {
    debugCallback?.call('Preprocessing image for recognition...');
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
    
    debugCallback?.call('Image preprocessing for recognition completed');
    return preprocessedData;
  }

  // Recognize text from preprocessed image
  Future<String> _recognizeText(Float32List preprocessedImage) async {
    try {
      final shape = [1, 3, OCRConstants.REC_TARGET_SIZE[0], OCRConstants.REC_TARGET_SIZE[1]];
      final inputOrt = OrtValueTensor.createTensorWithDataList(preprocessedImage, shape);
      final inputs = {'input': inputOrt};
      final runOptions = OrtRunOptions();

      final results = await recognitionModel?.runAsync(runOptions, inputs);
      inputOrt.release();
      runOptions.release();

      results?.forEach((element) {
        element?.release();
      });

      final output = results?.first.value;
      final logits = output as Float32List;
      final dims = output.dims;

      final batchSize = dims[0];
      final height = dims[1];
      final numClasses = dims[2];

      List<double> softmax(List<double> logits) {
        final expLogits = logits.map((x) => math.exp(x)).toList();
        final sumExpLogits = expLogits.reduce((a, b) => a + b);
        return expLogits.map((x) => x / sumExpLogits).toList();
      }

      final List<int> bestPath = [];
      for (int h = 0; h < height; h++) {
        final List<double> timestepLogits = logits.sublist(h * numClasses, (h + 1) * numClasses);
        final softmaxed = softmax(timestepLogits);
        final maxIndex = softmaxed.indexWhere((x) => x == softmaxed.reduce(math.max));
        bestPath.add(maxIndex);
      }

      final StringBuffer decodedText = StringBuffer();
      int prevIndex = -1;
      for (final index in bestPath) {
        if (index != numClasses - 1 && index != prevIndex) {
          decodedText.write(OCRConstants.VOCAB[index]);
        }
        prevIndex = index;
      }

      return decodedText.toString();
    } catch (e) {
      debugCallback?.call('Recognition error: $e');
      throw Exception('Recognition failed: $e');
    }
  }

  // Process image
  Future<List<OCRResult>> processImage(ui.Image image, {void Function(String)? debugCallback}) async {
    this.debugCallback = debugCallback;
    try {
      debugCallback?.call('Starting image processing...');
      
      final preprocessedImage = await _preprocessImageForDetection(image);
      debugCallback?.call('Image preprocessed for detection');
      
      final detectionResult = await _runDetection(preprocessedImage);
      debugCallback?.call('Detection completed');
      
      final boundingBoxes = await _extractBoundingBoxes(detectionResult);
      debugCallback?.call('Found ${boundingBoxes.length} bounding boxes');
      
      if (boundingBoxes.isEmpty) {
        return [];
      }

      final results = <OCRResult>[];
      for (var box in boundingBoxes) {
        final croppedImage = await _cropImage(image, box);
        final preprocessedCrop = await _preprocessImageForRecognition(croppedImage);
        final text = await _recognizeText(preprocessedCrop);
        if (text.isNotEmpty) {
          results.add(OCRResult(text: text, boundingBox: box));
        }
      }

      debugCallback?.call('Processed ${results.length} text regions');
      return results;
    } catch (e, stack) {
      debugCallback?.call('Error in processImage: $e\n$stack');
      throw Exception('Error in processImage: $e');
    }
  }

  // Crop image to bounding box
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
    return await picture.toImage(targetWidth.toInt(), targetHeight.toInt());
  }
}