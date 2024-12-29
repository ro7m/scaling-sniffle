import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:opencv_dart/opencv_dart.dart' as cv; 
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
      // 2. Create source Mat from heatmap bytes (RGBA)
      cv.Mat src = cv.Mat.create(
        rows: height,
        cols: width,
        type: cv.MatType.cv8UC4,
      );
      src.setDataWithBytes(heatmapBytes);

      // 3. Convert to grayscale
      cv.Mat gray = cv.Mat.zeros(height, width, cv.MatType.cv8UC1);
      cv.Imgproc.cvtColor(src, gray, cv.ColorConversionCodes.COLOR_RGBA2GRAY);

      // 4. Apply threshold
      cv.Mat binary = cv.Mat.zeros(height, width, cv.MatType.cv8UC1);
      cv.Imgproc.threshold(
        gray, 
        binary,
        77,  // threshold value
        255, // max value
        cv.ThresholdTypes.binary
      );

      // 5. Create kernel for morphological operation
      cv.Mat kernel = cv.Imgproc.getStructuringElement(
        cv.MorphShapes.rect,
        cv.Size(2, 2)
      );

      // 6. Apply morphological opening
      cv.Mat opened = cv.Mat.zeros(height, width, cv.MatType.cv8UC1);
      cv.Imgproc.morphologyEx(
        binary,
        opened,
        cv.MorphTypes.open,
        kernel
      );

      // 7. Find contours
      List<cv.Point> contours = [];
      cv.Mat hierarchy = cv.Mat.zeros(1, 1, cv.MatType.cv32SC4);
      
      cv.Imgproc.findContours(
        opened,
        contours,
        hierarchy,
        cv.RetrievalModes.external,
        cv.ContourApproximationModes.simple
      );

      // 8. Process contours and create bounding boxes
      List<BoundingBox> boundingBoxes = [];
      
      for (int i = 0; i < contours.length; i++) {
        cv.Rect boundRect = cv.Imgproc.boundingRect(contours[i]);
        
        if (boundRect.width > 2 && boundRect.height > 2) {
          // Create contour data for transformation
          Map<String, dynamic> contour = {
            'x': boundRect.x,
            'y': boundRect.y,
            'width': boundRect.width,
            'height': boundRect.height,
          };

          // Insert at the beginning (unshift equivalent)
          boundingBoxes.insert(0, 
            await transformBoundingBox(contour, boundingBoxes.length, [height, width])
          );
        }
      }

      // 9. Clean up OpenCV resources
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

  Future<Uint8List> createHeatmapFromProbMap(Float32List probMap, int width, int height) async {
    // Create RGBA byte array (4 bytes per pixel)
    final bytes = Uint8List(width * height * 4);
    
    for (int i = 0; i < probMap.length; i++) {
      final pixelValue = (probMap[i] * 255).round();
      final j = i * 4;
      bytes[j] = pixelValue;     // R
      bytes[j + 1] = pixelValue; // G
      bytes[j + 2] = pixelValue; // B
      bytes[j + 3] = 255;        // A
    }
    
    return bytes;
  }

  Future<BoundingBox> transformBoundingBox(Map<String, dynamic> contour, int id, List<int> size) async {
    double offset = (contour['width'] * contour['height'] * 1.8) / 
                   (2 * (contour['width'] + contour['height']));
    
    double p1 = clamp(contour['x'] - offset, size[1].toDouble()) - 1;
    double p2 = clamp(p1 + contour['width'] + 2 * offset, size[1].toDouble()) - 1;
    double p3 = clamp(contour['y'] - offset, size[0].toDouble()) - 1;
    double p4 = clamp(p3 + contour['height'] + 2 * offset, size[0].toDouble()) - 1;

    List<List<double>> coordinates = [
      [p1 / size[1], p3 / size[0]],
      [p2 / size[1], p3 / size[0]],
      [p2 / size[1], p4 / size[0]],
      [p1 / size[1], p4 / size[0]],
    ];

    return BoundingBox(
      id: id,
      x: coordinates[0][0],
      y: coordinates[0][1],
      width: coordinates[1][0] - coordinates[0][0],
      height: coordinates[2][1] - coordinates[0][1],
      coordinates: coordinates,
      config: {"stroke": getRandomColor()},
    );
  }

  double clamp(double value, double max) {
    return math.max(0, math.min(value, max));
  }

  String getRandomColor() {
    return '#${(math.Random().nextDouble() * 0xFFFFFF).toInt().toRadixString(16).padLeft(6, '0')}';
  }
}