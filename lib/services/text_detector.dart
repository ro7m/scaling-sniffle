import 'package:image/image.dart' as img;
import 'dart:ui' as ui;
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

    return Float32List.fromList(flattenedOutput);
  }

  Future<List<BoundingBox>> processDetectionOutput(Float32List probMap) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];
    
    // 1. Convert probability map to heatmap image
    final heatmapImage = await createHeatmapFromProbMap(probMap, width, height);
    
    // 2. Process the heatmap to extract bounding boxes
    return extractBoundingBoxesFromHeatmap(heatmapImage, width, height);
  }

  Future<img.Image> createHeatmapFromProbMap(Float32List probMap, int width, int height) {
    // Create a grayscale image from probability map
    final image = img.Image(width: width, height: height);
    
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final prob = probMap[y * width + x];
        final pixelValue = (prob * 255).round();
        image.setPixel(x, y, img.ColorRgb8(pixelValue, pixelValue, pixelValue));
      }
    }
    
    return Future.value(image);
  }

  Future<List<BoundingBox>> extractBoundingBoxesFromHeatmap(img.Image heatmap, int width, int height) async {
    // 1. Convert to grayscale if not already
    final grayscale = img.grayscale(heatmap);
    
    // 2. Apply threshold (similar to cv.threshold in JS version)
    final binary = img.binarize(grayscale, threshold: 77);
    
    // 3. Apply morphological opening (similar to cv.morphologyEx)
    final opened = img.morphologyEx(
      binary,
      img.kernelCross(2), // 2x2 kernel
      img.MorphologyOperation.open
    );
    
    // 4. Find contours (we'll use connected components as an alternative)
    final List<BoundingBox> boundingBoxes = [];
    final visited = List.generate(height, (_) => List.filled(width, false));
    
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (!visited[y][x] && opened.getPixel(x, y) == img.ColorRgb8(255, 255, 255)) {
          final contour = _findContour(opened, x, y, visited);
          
          // Calculate bounding box for the contour
          if (contour.isNotEmpty) {
            int minX = width, minY = height, maxX = 0, maxY = 0;
            
            for (final point in contour) {
              minX = math.min(minX, point.x);
              minY = math.min(minY, point.y);
              maxX = math.max(maxX, point.x);
              maxY = math.max(maxY, point.y);
            }
            
            final boxWidth = maxX - minX;
            final boxHeight = maxY - minY;
            
            if (boxWidth > 2 && boxHeight > 2) {
              boundingBoxes.add(BoundingBox(
                x: minX / width,
                y: minY / height,
                width: boxWidth / width,
                height: boxHeight / height,
              ));
            }
          }
        }
      }
    }
    
    return boundingBoxes;
  }

  List<Point<int>> _findContour(img.Image image, int startX, int startY, List<List<bool>> visited) {
    final List<Point<int>> contour = [];
    final queue = Queue<Point<int>>();
    queue.add(Point(startX, startY));
    
    while (queue.isNotEmpty) {
      final point = queue.removeFirst();
      final x = point.x;
      final y = point.y;
      
      if (x >= 0 && x < image.width && y >= 0 && y < image.height && 
          !visited[y][x] && image.getPixel(x, y) == img.ColorRgb8(255, 255, 255)) {
        visited[y][x] = true;
        contour.add(Point(x, y));
        
        // Add neighbors
        queue.add(Point(x + 1, y));
        queue.add(Point(x - 1, y));
        queue.add(Point(x, y + 1));
        queue.add(Point(x, y - 1));
      }
    }
    
    return contour;
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

  List<int> postprocessProbabilityMap(Float32List probMap) {
    final threshold = 0.1;
    return probMap.map((prob) => prob > threshold ? 1 : 0).toList();
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