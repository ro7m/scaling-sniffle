import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart';
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
      // Create Mat from bytes
      Mat src = Mat.fromBytes(
        size[0], // height
        size[1], // width
        MatType.CV_8UC1,
        heatmapBytes
      );

      // Create output Mat for threshold
      Mat thresholded = new Mat();
      
      // Apply threshold
      Core.threshold(
        src,
        thresholded,
        77,
        255,
        Core.THRESH_BINARY
      );

      // Create structuring element
      Mat kernel = Imgproc.getStructuringElement(
        Imgproc.MORPH_RECT,
        Size(2, 2)
      );

      // Create output Mat for morphology
      Mat opened = new Mat();
      
      // Apply morphological opening
      Imgproc.morphologyEx(
        thresholded,
        opened,
        Imgproc.MORPH_OPEN,
        kernel
      );

      // Find contours
      List<List<Point>> contours = [];
      Mat hierarchy = new Mat();
      
      Imgproc.findContours(
        opened.clone(),
        contours,
        hierarchy,
        Imgproc.RETR_EXTERNAL,
        Imgproc.CHAIN_APPROX_SIMPLE
      );

      final List<BoundingBox> boundingBoxes = [];

      // Process contours
      for (var contour in contours) {
        Rect rect = Imgproc.boundingRect(contour);
        
        if (rect.width > 2 && rect.height > 2) {
          // Insert at the beginning of the list (equivalent to unshift)
          boundingBoxes.insert(0, await transformBoundingBox({
            'x': rect.x,
            'y': rect.y,
            'width': rect.width,
            'height': rect.height
          }, boundingBoxes.length, size));
        }
      }

      // Clean up resources
      src.release();
      thresholded.release();
      kernel.release();
      opened.release();
      hierarchy.release();

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