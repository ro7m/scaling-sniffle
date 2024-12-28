import '../models/bounding_box.dart';

class OCRResult {
  final String text;
  final BoundingBox boundingBox;

  OCRResult({required this.text, required this.boundingBox});
}