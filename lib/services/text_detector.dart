import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:dartcv4/dartcv.dart' as cv;
import 'package:onnxruntime/onnxruntime.dart';
import '../models/bounding_box.dart';
import '../constants.dart';

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
    return maxX >= 0 && maxY >= 0 && maxX > minX && maxY > minY;
  }
}

class TextDetector {
  final OrtSession detectionModel;

  TextDetector(this.detectionModel);

  Future<Float32List> runDetection(Float32List preprocessedImage) async {
    final shape = [1, 3, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]];
    final inputOrt = OrtValueTensor.createTensorWithDataList(preprocessedImage, shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();

    final results = await detectionModel.runAsync(runOptions, inputs);
    inputOrt.release();
    runOptions.release();

    final output = results?.first?.value;
    if (output == null) {
      throw Exception('Detection model output is null');
    }
    final flattenedOutput = _flattenNestedList(output as List);

    results?.forEach((element) {
      element?.release();
    });

    return _convertToFloat32ListAndApplySigmoid(flattenedOutput);
  }

  Float32List _flattenNestedList(List nestedList) {
    final List<double> flattened = [];
    void flatten(dynamic item) {
      if (item is List) {
        for (var subItem in item) {
          flatten(subItem);
        }
      } else if (item is num) {
        flattened.add(item.toDouble());
      }
    }
    flatten(nestedList);
    return Float32List.fromList(flattened);
  }

  Float32List _convertToFloat32ListAndApplySigmoid(Float32List flattened) {
    final Float32List result = Float32List(flattened.length);
    for (int i = 0; i < flattened.length; i++) {
      result[i] = 1.0 / (1.0 + math.exp(-flattened[i]));
    }
    return result;
  }





  Float32List postprocessProbabilityMap(Float32List probMap) {
    const threshold = 0.1;
    return Float32List.fromList(
      probMap.map((prob) => prob > threshold ? 1.0 : 0.0).toList()
    );
  }

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

  Future<List<BoundingBox>> processImage(Float32List probMap) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];

    // 1. Create heatmap from probability map
    final heatmapBytes = await createHeatmapFromProbMap(probMap, width, height);
    
    try {
      // Fix #1: Create source Mat with correct type
      cv.Mat src = cv.Mat.create(
        rows: height,
        cols: width,
        type: cv.MatType.cvType(8, 4),  // 8-bit 4-channel
      );
      src.setDataWithBytes(heatmapBytes);

      // Fix #2: Create grayscale Mat
      cv.Mat gray = cv.Mat.zeros(height, width, cv.MatType.cvType(8, 1));  // 8-bit 1-channel

      // Fix #3, #4: Convert to grayscale using correct cvtColor syntax
      cv.cvtColor(src, dst: gray, code: cv.ColorConversionCode.cvColorRgbaGray);

      // Fix #5: Create binary Mat
      cv.Mat binary = cv.Mat.zeros(height, width, cv.MatType.cvType(8, 1));

      // Fix #6, #7: Apply threshold with correct parameters
      cv.threshold(
        gray,
        dst: binary,
        thresh: 77,
        maxval: 255,
        type: cv.ThresholdType.cvThreshBinary,
      );

      // Fix #8: Create kernel for morphological operation
      cv.Mat kernel = cv.getStructuringElement(
        shape: cv.MorphShape.cvShapeRect,
        size: cv.Size(2, 2)
      );

      // Fix #9, #10, #11: Apply morphological opening
      cv.Mat opened = cv.Mat.zeros(height, width, cv.MatType.cvType(8, 1));
      cv.morphologyEx(
        binary,
        dst: opened,
        op: cv.MorphType.cvMorphOpen,
        kernel: kernel
      );

      // Find contours - using updated API
      List<List<cv.Point>> contours = [];
      cv.Mat hierarchy = cv.Mat.zeros(1, 1, cv.MatType.cvType(32, 1));
      
      cv.findContours(
        opened,
        contours: contours,
        hierarchy: hierarchy,
        mode: cv.ContourRetrievalMode.cvRetrExternal,
        method: cv.ContourApproximationMode.cvChainApproxSimple
      );

      // Process contours and create bounding boxes
      List<BoundingBox> boundingBoxes = [];
      
      for (var contour in contours) {
        cv.Rect boundRect = cv.boundingRect(contour);
        
        if (boundRect.width > 2 && boundRect.height > 2) {
          Map<String, dynamic> contourData = {
            'x': boundRect.x,
            'y': boundRect.y,
            'width': boundRect.width,
            'height': boundRect.height,
          };

          boundingBoxes.insert(0, 
            await transformBoundingBox(contourData, boundingBoxes.length, [height, width])
          );
        }
      }

      // Clean up OpenCV resources
      src.release();
      gray.release();
      binary.release();
      kernel.release();
      opened.release();
      hierarchy.release();

      return boundingBoxes;

    } catch (e) {
      print('OpenCV Error: $e');
      rethrow;
    }
  }

  double clamp(double value, double max) {
    return math.max(0, math.min(value, max));
  }

  String getRandomColor() {
    return '#${(math.Random().nextDouble() * 0xFFFFFF).toInt().toRadixString(16).padLeft(6, '0')}';
  }

}