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
      cv.Mat src = cv.Mat.create(
        size[0],  // height
        size[1],  // width
        cv.MatType.cv8UC1
      );
      src.setDataWithBytes(heatmapBytes);

      // Create destination matrix for threshold
      cv.Mat thresholded = cv.Mat.create(size[0], size[1], cv.MatType.cv8UC1);
      
      // Apply threshold
      double thresh = cv.Imgproc.threshold(
        src,                    // src
        thresholded,           // dst
        77,                    // thresh value
        255,                   // maxval
        cv.ThresholdTypes.binary // threshold type
      );

      // Create structuring element
      cv.Mat kernel = cv.Imgproc.getStructuringElement(
        cv.MorphShapes.rect,   // shape
        cv.Size(2, 2)         // kernel size
      );

      // Create output Mat for morphology
      cv.Mat opened = cv.Mat.create(size[0], size[1], cv.MatType.cv8UC1);
      
      // Apply morphological opening
      cv.Imgproc.morphologyEx(
        thresholded,           // src
        opened,                // dst
        cv.MorphTypes.open,    // operation type
        kernel                 // kernel
      );

      // Find contours
      List<cv.MatVector> contours = [];
      cv.Mat hierarchy = cv.Mat.create(1, 1, cv.MatType.cv32SC4);
      
      cv.Imgproc.findContours(
        opened,                              // image
        contours,                            // contours
        hierarchy,                           // hierarchy
        cv.RetrievalModes.external,         // mode
        cv.ContourApproximationModes.simple // method
      );

      final List<BoundingBox> boundingBoxes = [];

      // Process contours
      for (var contour in contours) {
        cv.Rect rect = cv.Imgproc.boundingRect(contour);
        
        // Access rect properties directly using dot notation
        if (rect.width > 2 && rect.height > 2) {
          boundingBoxes.insert(0, await transformBoundingBox(
            // Create a map with the rect properties
            {
              'x': rect.x,
              'y': rect.y,
              'width': rect.width,
              'height': rect.height
            }, 
            boundingBoxes.length, 
            size
          ));
        }
      }

      // Clean up resources
      src.release();
      thresholded.release();
      kernel.release();
      opened.release();
      hierarchy.release();
      for (var contour in contours) {
        contour.release();
      }

      return boundingBoxes;
    } catch (e, stackTrace) {
      print('OpenCV Error: $e');
      print('Stack trace: $stackTrace');
      throw Exception('Failed to process image with OpenCV: $e');
    }
  }

  Future<BoundingBox> transformBoundingBox(Map<String, dynamic> rectData, int id, List<int> size) async {
    double offset = (rectData['width'] * rectData['height'] * 1.8) / 
                   (2 * (rectData['width'] + rectData['height']));
    
    double p1 = clamp(rectData['x'] - offset, size[1].toDouble()) - 1;
    double p2 = clamp(p1 + rectData['width'] + 2 * offset, size[1].toDouble()) - 1;
    double p3 = clamp(rectData['y'] - offset, size[0].toDouble()) - 1;
    double p4 = clamp(p3 + rectData['height'] + 2 * offset, size[0].toDouble()) - 1;

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

  // Add color generation similar to JavaScript
  String getRandomColor() {
    final random = math.Random();
    return '#${(random.nextDouble() * 0xFFFFFF).toInt().toRadixString(16).padLeft(6, '0')}';
  }

  double clamp(double number, double size) {
    return math.max(0, math.min(number, size));
  }
}