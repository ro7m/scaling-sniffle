import 'dart:typed_data';
import 'dart:math' as math;
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

    return createHeatmapFromProbMap(flattenedOutput, OCRConstants.TARGET_SIZE[0], OCRConstants.TARGET_SIZE[1]);
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

  Future<List<BoundingBox>> extractBoundingBoxes(Float32List probMap, {void Function(String)? debugCallback}) async {
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
}