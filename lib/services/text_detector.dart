import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
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
    final flattenedOutput = _flattenNestedList(output as List);

    results?.forEach((element) {
      element?.release();
    });

    return Float32List.fromList(flattenedOutput);
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

  Future<List<BoundingBox>> processDetectionOutput(Float32List probMap) async {
    final width = OCRConstants.TARGET_SIZE[0];
    final height = OCRConstants.TARGET_SIZE[1];
    
    final heatmapImage = await createHeatmapFromProbMap(probMap, width, height);
    return extractBoundingBoxesFromHeatmap(heatmapImage, [width, height]);
  }

  Future<Uint8List> createHeatmapFromProbMap(Float32List probMap, int width, int height) async {
    final imageBytes = Uint8List(width * height);
    
    for (int i = 0; i < width * height; i++) {
      imageBytes[i] = (probMap[i] * 255).round();
    }
    
    return imageBytes;
  }

  Future<List<BoundingBox>> extractBoundingBoxesFromHeatmap(Uint8List heatmapBytes, List<int> size) async {
    try {
      // Create the image matrix from bytes
      final Map<String, dynamic> input = {
        "data": heatmapBytes,
        "width": size[1],
        "height": size[0],
      };

      // Apply threshold
      final thresholded = await cv.Cv2.threshold(input, {"thresh": 77, "maxval": 255});

      // Apply morphological opening
      final morphKernel = await cv.Cv2.getStructuringElement({
        "shape": cv.Cv2.MORPH_RECT,
        "ksize": [2, 2]
      });

      final opened = await cv.Cv2.morphologyEx({
        "src": thresholded,
        "op": cv.Cv2.MORPH_OPEN,
        "kernel": morphKernel
      });

      // Find contours
      final contours = await cv.Cv2.findContours({
        "image": opened,
        "mode": cv.Cv2.RETR_EXTERNAL,
        "method": cv.Cv2.CHAIN_APPROX_SIMPLE,
      });

      final List<BoundingBox> boundingBoxes = [];

      // Process contours and use unshift (add at beginning) like in JavaScript
      for (final contour in contoursResult['contours']) {
        final rect = await cv.Cv2.boundingRect({"points": contour});
        
        if (rect['width'] > 2 && rect['height'] > 2) {
          // Insert at the beginning of the list (equivalent to unshift)
          boundingBoxes.insert(0, await transformBoundingBox(rect, boundingBoxes.length, size));
        }
      }

      return boundingBoxes;
    } catch (e) {
      print('OpenCV Error: $e');
      throw Exception('Failed to process image with OpenCV: $e');
    }
  }


    Future<BoundingBox> transformBoundingBox(Map<String, dynamic> contour, int id, List<int> size) async {
    double offset = (contour['width'] * contour['height'] * 1.8) / 
                   (2 * (contour['width'] + contour['height']));
    
    // Match exactly with JavaScript implementation
    double p1 = clamp(contour['x'] - offset, size[1].toDouble()) - 1;
    double p2 = clamp(p1 + contour['width'] + 2 * offset, size[1].toDouble()) - 1;
    double p3 = clamp(contour['y'] - offset, size[0].toDouble()) - 1;
    double p4 = clamp(p3 + contour['height'] + 2 * offset, size[0].toDouble()) - 1;

    // Create coordinates array exactly like JavaScript version
    List<List<double>> coordinates = [
      [p1 / size[1], p3 / size[0]],
      [p2 / size[1], p3 / size[0]],
      [p2 / size[1], p4 / size[0]],
      [p1 / size[1], p4 / size[0]],
    ];

    // Let's modify our BoundingBox class to include all the JavaScript properties
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

  // Add color generation similar to JavaScript
  String getRandomColor() {
    final random = math.Random();
    return '#${(random.nextDouble() * 0xFFFFFF).toInt().toRadixString(16).padLeft(6, '0')}';
  }

  double clamp(double number, double size) {
    return math.max(0, math.min(number, size));
  }
}