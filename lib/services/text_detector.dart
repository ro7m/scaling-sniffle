import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:collection';
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

class Point {
  final int x;
  final int y;
  Point(this.x, this.y);
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

  void _floodFill(int x, int y, int label, List<List<int>> labels, List<List<bool>> binaryMap, _BBox bbox) {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];
    
    final queue = Queue<Point>();
    queue.add(Point(x, y));

    while (queue.isNotEmpty) {
      final point = queue.removeFirst();
      final px = point.x;
      final py = point.y;

      if (px >= 0 && px < width && py >= 0 && py < height && 
          binaryMap[py][px] && labels[py][px] == -1) {
        labels[py][px] = label;
        bbox.update(px, py);

        // Add neighbors in a more efficient order
        if (px + 1 < width) queue.add(Point(px + 1, py));
        if (px - 1 >= 0) queue.add(Point(px - 1, py));
        if (py + 1 < height) queue.add(Point(px, py + 1));
        if (py - 1 >= 0) queue.add(Point(px, py - 1));
      }
    }
  }

  Future<List<BoundingBox>> processImage(Float32List detectionResult) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];

    // 1. Create heatmap from detection result (probability map)
    final heatmapBytes = await createHeatmapFromProbMap(detectionResult, width, height);
    
    // 2. Create binary map from heatmap
    List<List<bool>> binaryMap = List.generate(
      height,
      (y) => List.generate(
        width,
        (x) => heatmapBytes[4 * (y * width + x)] > 77  // Using R channel value and threshold of 77
      ),
    );

    // 3. Extract bounding boxes using flood fill
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

    // 4. Transform bounding boxes
    List<BoundingBox> boundingBoxes = [];
    for (var entry in components.entries) {
      var component = entry.value;
      final boxWidth = component.maxX - component.minX + 1;
      final boxHeight = component.maxY - component.minY + 1;

      if (boxWidth > 2 && boxHeight > 2) {
        Map<String, dynamic> contour = {
          'x': component.minX,
          'y': component.minY,
          'width': boxWidth,
          'height': boxHeight,
        };
        
        boundingBoxes.insert(0, 
          await transformBoundingBox(contour, boundingBoxes.length, [height, width])
        );
      }
    }

    return boundingBoxes;
  }

  Future<Uint8List> createHeatmapFromProbMap(Float32List probMap, int width, int height) async {
    // Create RGBA byte array (4 bytes per pixel)
    final bytes = Uint8List(width * height * 4);
    
    for (int i = 0; i < probMap.length; i++) {
      // Convert probability to pixel value (0-255)
      final pixelValue = (probMap[i] * 255).round();
      final j = i * 4;
      
      // Set all channels to the same value for grayscale effect
      bytes[j] = pixelValue;     // R
      bytes[j + 1] = pixelValue; // G
      bytes[j + 2] = pixelValue; // B
      bytes[j + 3] = 255;        // A (fully opaque)
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