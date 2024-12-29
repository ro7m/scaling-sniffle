import 'dart:typed_data';
import 'dart:math' as math;
import 'package:onnxruntime/onnxruntime.dart';
import '../models/bounding_box.dart';
import '../constants.dart';

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
    
    // Convert nested list to flat Float32List
    final flattenedOutput = _flattenNestedList(output as List);
    
    // Apply accurate sigmoid
    final Float32List sigmoidOutput = Float32List(flattenedOutput.length);
    for (int i = 0; i < flattenedOutput.length; i++) {
      sigmoidOutput[i] = 1.0 / (1.0 + math.exp(-flattenedOutput[i]));
    }

    results?.forEach((element) {
      element?.release();
    });

    return sigmoidOutput;
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

  Future<List<BoundingBox>> processImage(Float32List detectionResult) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];

    // Create binary map with proper thresholding
    final List<List<bool>> binaryMap = List.generate(
      height,
      (y) => List.generate(
        width,
        (x) => detectionResult[y * width + x] > 0.3 // Adjusted threshold
      ),
    );

    // Apply morphological operations (simple dilation)
    final dilatedMap = _dilate(binaryMap);
    
    // Find connected components
    final components = _findConnectedComponents(dilatedMap);

    // Convert components to bounding boxes
    final boundingBoxes = <BoundingBox>[];
    for (final component in components) {
      if (_isValidComponent(component)) {
        final box = await _createBoundingBox(component, boundingBoxes.length, [height, width]);
        boundingBoxes.add(box);
      }
    }

    return boundingBoxes;
  }

  List<List<bool>> _dilate(List<List<bool>> input) {
    final height = input.length;
    final width = input[0].length;
    final output = List.generate(
      height,
      (y) => List.generate(width, (x) => false),
    );

    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        output[y][x] = input[y][x] ||
            input[y - 1][x] ||
            input[y + 1][x] ||
            input[y][x - 1] ||
            input[y][x + 1];
      }
    }
    return output;
  }

  List<_Component> _findConnectedComponents(List<List<bool>> binaryMap) {
    final height = binaryMap.length;
    final width = binaryMap[0].length;
    final visited = List.generate(
      height,
      (y) => List.generate(width, (x) => false),
    );
    final components = <_Component>[];

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (binaryMap[y][x] && !visited[y][x]) {
          final component = _Component();
          _floodFill(x, y, binaryMap, visited, component);
          components.add(component);
        }
      }
    }

    return components;
  }

  void _floodFill(int x, int y, List<List<bool>> binaryMap, List<List<bool>> visited, _Component component) {
    final queue = <Point>[];
    queue.add(Point(x, y));

    while (queue.isNotEmpty) {
      final point = queue.removeAt(0);
      final px = point.x;
      final py = point.y;

      if (px >= 0 && px < binaryMap[0].length && py >= 0 && py < binaryMap.length &&
          binaryMap[py][px] && !visited[py][px]) {
        visited[py][px] = true;
        component.addPoint(px, py);

        queue.add(Point(px + 1, py));
        queue.add(Point(px - 1, py));
        queue.add(Point(px, py + 1));
        queue.add(Point(px, py - 1));
      }
    }
  }

  bool _isValidComponent(_Component component) {
    final width = component.maxX - component.minX + 1;
    final height = component.maxY - component.minY + 1;
    final area = width * height;
    
    // Filter out components that are too small or have invalid aspect ratios
    return width > 4 && height > 4 && area > 64 && width / height < 10 && height / width < 10;
  }

  Future<BoundingBox> _createBoundingBox(_Component component, int id, List<int> size) async {
    final width = component.maxX - component.minX + 1;
    final height = component.maxY - component.minY + 1;
    
    // Calculate padding based on component size
    final padding = (width * height * 0.15) / (2 * (width + height));
    
    final x1 = _clamp((component.minX - padding) / size[1], 0.0, 1.0);
    final x2 = _clamp((component.maxX + padding) / size[1], 0.0, 1.0);
    final y1 = _clamp((component.minY - padding) / size[0], 0.0, 1.0);
    final y2 = _clamp((component.maxY + padding) / size[0], 0.0, 1.0);

    return BoundingBox(
      id: id,
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1,
      coordinates: [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
      ],
      config: {"stroke": _getRandomColor()},
    );
  }

  double _clamp(double value, double min, double max) {
    return math.max(min, math.min(value, max));
  }

  String _getRandomColor() {
    return '#${(math.Random().nextDouble() * 0xFFFFFF).toInt().toRadixString(16).padLeft(6, '0')}';
  }
}

class _Component {
  int minX = 999999, minY = 999999;
  int maxX = -1, maxY = -1;

  void addPoint(int x, int y) {
    minX = math.min(minX, x);
    minY = math.min(minY, y);
    maxX = math.max(maxX, x);
    maxY = math.max(maxY, y);
  }
}

class Point {
  final int x;
  final int y;
  Point(this.x, this.y);
}
